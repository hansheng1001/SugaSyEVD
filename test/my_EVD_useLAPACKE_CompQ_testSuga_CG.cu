
#include <cstring>
#include <iomanip> // 包含操纵器的头文件
#include <iostream>
// #include <string>
// #include <vector>
// #include <lapacke.h>
// #include <mkl_lapacke.h>
#include <cusolverDn.h>
#include <algorithm> // std::sort


// 如果使用NVCC直接编译设置为1
// 命令 nvcc my_EVD_useLAPACKE_CompQ.cu -o 
// my_EVD_useLAPACKE_CompQ -lcublas -lcusolver --gpu-max-threads-per-block=1024
// 直接执行./my_EVD_useLAPACKE_CompQ 128 32 128

// 使用cmake编译,这设置为0
// 直接在根目录下使用cmake命令
// cd SugaSyEVD_CG_Test
// rm build -rf
// source env.sh
// cmake -Bbuild -S .
// cmake --build build -j
// cd build/src/EVD
// 执行命令：./my_EVD_useLAPACKE_CompQ 128 32 128

#define USE_NVCC_COMPILE_ENABLE 0
#if USE_NVCC_COMPILE_ENABLE
#include <cooperative_groups.h>
// #include <string>
#include <vector>
#include <fstream>
#include <sstream>

#else
#include "bugle_chasingV10.h"
#include "fileOpTool.h"
#include "kernelOther.h"
#include "myBase.h"
#endif

using namespace std;

constexpr int DIM_X = 16;
constexpr int DIM_Y = 16;


#if USE_NVCC_COMPILE_ENABLE
namespace cg = cooperative_groups;

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

 
static cudaEvent_t start, stop;
static void startTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

static float stopTimer()
{
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

template <typename T>
void printDeviceMatrixV2(T *dA, long ldA, long rows, long cols)
{
  T matrix;

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {
      cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);
      // printf("%f ", matrix[i * cols + j]);//按行存储优先
      // printf("%10.4f", matrix); // 按列存储优先
      // printf("%12.6f", matrix); // 按列存储优先
      // printf("%.20f ", matrix); // 按列存储优先
      printf("%.14f ", matrix); // 按列存储优先
    }
    printf("\n");
  }
}

template void printDeviceMatrixV2(double *dA, long ldA, long rows, long cols);
template void printDeviceMatrixV2(float *dA, long ldA, long rows, long cols);


// #include "fileOpTool.h"

static __inline__ __device__ double warpAllReduceSum(double val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}


__device__ int g_overFlag = 0;


template <int BandWidth>
// __device__ void chasing_kernel_one_timeV7(int n, int b, double *subA, int ldSubA, int startRow)
__global__ void chasing_kernel_one_timeV10(int n,
                                           int b,
                                           double *subA,
                                           int ldSubA,
                                           double *dU,
                                           long ldU,
                                           int blockNum,
                                           int *com
                                          //  int *g_overFlag
                                          )
{
  auto grid  = cg::this_grid();
  // auto block = cg::this_thread_block();

  // 错误检查
  int Nx = blockDim.x;
  int Ny = blockDim.y;

  // 内部使用,不进行判断
  if (BandWidth != b)
  {
    return;
  }

  int warpGroupThdCount = Ny / 2;

  int bInx = grid.block_rank();

  int i = threadIdx.x;
  int j = threadIdx.y;

  __shared__ double u[BandWidth];

  __shared__ double S1[BandWidth * BandWidth]; // 共享内存,用于存放[B1,S,B2]进行变换
  __shared__ double S2[BandWidth * BandWidth]; // 共享内存,用于存放[B1,S,B2]进行变换
  int ldSS = b;

  double nu;
  double utx;

  for (; 0 == g_overFlag; bInx += blockNum)
  {
    int opColB1;
    int opRowB1;

    // 求出u的所在的列数
    double *uB = dU + bInx * ldU;

    long opRow = bInx + 1;

    int rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
    int colB1 = 1;

    // 找到B1在SubA中的位置
    // 所在列数为opRow-colB1,所在行数为colB1
    double *B1 = subA + colB1 + (opRow - colB1) * ldSubA;

    bool firstFlag = true;
    bool cycFlag   = true;

    // 最开始的时候S没有数据
    int rowS;
    int colS;

    double *S; // 初始的时候因为不会真正的使用它,所以不赋值没问题

    while (cycFlag && 0 == g_overFlag)
    {
      if ((bInx < n - 2) && (false == ((0 != bInx) && (opRow + 2 * b > com[bInx - 1]))))
      {

        if (0 != j)
        {
          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;

        }
        else
        {
          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;
        }

        __syncthreads();

        if (j < warpGroupThdCount)
        {
          opRow += rowB1;                   // 更新下一次A的长度
          rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

        }
        else
        {
          // 这里面的线程也需要更新这些局部变量
          opRow += rowB1;                   // 更新下一次A的长度
          rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

        }

        // 判断退出条件
        if (rowB1 <= 1)
        {

          if ((n - 3) == bInx && 0 == threadIdx.x && 0 == threadIdx.y)
          {

            g_overFlag = 1;
          }
          __syncthreads();


          cycFlag = false;
        }

        // 需要写入同步条件
        if ((0 == i) && (0 == j))
        {
          com[bInx] = opRow;

          if (false == cycFlag)
          {
            com[bInx] = n + 3 * b;
          }
        }

        __syncthreads();
      }

      grid.sync();
    }
  }
}

#endif

// 严格使用对称性
// 创建(b+1)*n个线程,每个线程处理1个元素
static __global__ void
kernel_bugle_chasing_cpydA2dSubA(int n, int b, double *dA, long ldA, double *dSubA, int ldSubA)
{

  // int bInx = blockIdx.y * gridDim.x + blockIdx.x;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // 伪代码
  // j索引等于本线程复制的A的列索引
  //

  // 起始位置和结束位置都是在同列中,计算的是行数
  // 1.找到A的起始复制位置--就是j

  // if (i < (b + 1) && j < n)
  if ((i < 2 * b) && j < n)
  {
    // 2.找到A的结束位置
    // int end = min(n, j + 2b);
    int end = min(n, j + b + 1); // 开始时,下面的b-1个元素是0,不用进行拷贝

    // 3.计算复制个数
    int count = end - j;

    // printf("block[%d] [%d][%d] come line=%d,count=%d.\n", bInx, i, j, __LINE__, count);

    if (i < count)
    {
      dSubA[i + j * ldSubA] = dA[j + i + j * ldA];
    }
    else
    {
      dSubA[i + j * ldSubA] = 0.0;
    }
  }
}

// 只需要拷贝主对角线和副对角线元素
// 创建n个线程,每个线程处理2个元素
static __global__ void
kernel_bugle_chasing_cpydSubA2dA(int n, int b, double *dSubA, int ldSubA, double *dA, long ldA)
{
  // int i = bInx * blockDim.x + threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // 伪代码
  // j索引等于本线程复制的A的列索引

  // 1.找到SubA的起始位置
  // 就是0

  // 2.拷贝数据
  if (i < n)
  {
    dA[i + i * ldA] = dSubA[i * ldSubA];

    if (i < n - 1)
    {
      dA[i + 1 + i * ldA] = dSubA[i * ldSubA + 1];

      // 增加部分,用于对后向误差进行求解
      dA[i + (i + 1) * ldA] = dSubA[i * ldSubA + 1];
    }
  }
}

std::vector<std::vector<double>> readMatrixFromFile(const std::string &fileName)
{
  std::vector<std::vector<double>> matrix;

  std::ifstream file(fileName);

  if (file.is_open())
  {
    std::string line;
    while (getline(file, line))
    {
      std::vector<double> row;
      std::stringstream ss(line);
      std::string cell;

      while (getline(ss, cell, ','))
      {
        row.push_back(std::stod(cell));
      }

      matrix.push_back(row);
    }

    file.close();
    std::cout << "Matrix read from " << fileName << std::endl;
  }
  else
  {
    std::cout << "Failed to open file: " << fileName << std::endl;
  }

  return matrix;
}

void fillMatrix(double *matrix, std::vector<std::vector<double>> &data)
{
  long rows = data.size();
  long cols = data[0].size();

  // 这是行优先(row-major order, RMO)的存储方式
  // for (long i = 0; i < rows; i++) {
  //   for (long j = 0; j < cols; j++) {
  //     matrix[i * cols + j] = data[i][j];
  //   }
  // }

  for (long i = 0; i < cols; i++)
  {
    for (long j = 0; j < rows; j++)
    {
      matrix[i * rows + j] = data[j][i];
    }
  }
}

int main(int argc, char *argv[])
{
  // 要求nb必须小于n，且n必须是nb的倍数
  long m, n;
  long b = 32;

  long nb = 4 * b;

  if (4 != argc)
  {
    cout << "Usage(b = nb in ZY): AppName <n> <b> <nb>" << endl;
    return 0;
  }

  m = n = atol(argv[1]);
  b     = atol(argv[2]);
  nb    = atol(argv[3]);



  cout << "SyEVD use ZY_ZY:" << endl;



  cusolverDnHandle_t cusolver_handle;
  cublasHandle_t cublas_handle;

  cusolverDnCreate(&cusolver_handle);
  cublasCreate(&cublas_handle);


  double *dA;
  CUDA_RT_CALL(cudaMalloc(&dA, sizeof(double) * m * n));

  dim3 gridDim((m + DIM_X - 1) / DIM_X, (n + DIM_Y - 1 ) / DIM_Y);
  dim3 blockDim(DIM_X, DIM_Y);

  string fileName = "/work/home/szhang94/wanghs/SugaSyEVD/data/Symatrix_A_128x128.csv";
  vector<vector<double>> data = readMatrixFromFile(fileName);

  m = data.size();
  n = data[0].size();

  cout << "n=" << n << ", b=" << b << ", nb=" << nb << endl;

  double *A;

  A = (double *)malloc(sizeof(double) * m * n);

  fillMatrix(A, data);
  cudaMemcpy(dA, A, sizeof(double) * m * n, cudaMemcpyHostToDevice);
  free(A);

  int *info;
  CUDA_RT_CALL(cudaMalloc(&info, sizeof(int)));

  CUDA_RT_CALL(cudaGetLastError());

  long  ldA;
  ldA      = m;

  printf("Begin dA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dA, ldA, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3); 

  double *dSubA;
  cudaMalloc(&dSubA, sizeof(double) * (2 * b) * n);
  int ldSubA = 2 * b;


  CUDA_RT_CALL(cudaGetLastError());
  cudaDeviceSynchronize();

  int dev                = 0;

  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

  printf("Device %d supports cooperative launch: %s\n", dev, supportsCoopLaunch ? "true" : "false");

  // 这个就是最多创建的块数
  /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel
  /// arguments
  int numBlocksPerSm = 0;
  // Number of threads my_kernel will be launched with
  int numThreads = 64 * 16;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);


  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      chasing_kernel_one_timeV10<32>,
      numThreads,
      0);
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;


  // launch
  // int blockNum = deviceProp.multiProcessorCount;

  printf("maxThreadsPerBlock: %d\n", deviceProp.maxThreadsPerBlock);
  printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
  printf("numBlocksPerSm: %d\n", numBlocksPerSm);

  // printf("numBlocksPerSm: %d\n", numBlocksPerSm);
  printf("blockNum: %d\n", blockNum);


  startTimer();


  // 将矩阵复制到条带化矩阵中
  dim3 blockDimcpydA2dSubA(DIM_X, DIM_Y);
  dim3 gridDimcpydA2dSubA((2 * b + DIM_X-1) / DIM_X, (n + DIM_Y - 1) / DIM_Y);
  kernel_bugle_chasing_cpydA2dSubA<<<gridDimcpydA2dSubA, blockDimcpydA2dSubA>>>(n,
                                                                                b,
                                                                                dA,
                                                                                ldA,
                                                                                dSubA,
                                                                                ldSubA);


  double *dU;
  cudaMalloc(&dU, sizeof(double) * m * n);
  long ldU = m;

  // launchKernel_ClearMatrix(gridDim, blockDim, m, n, dU, ldU);
  cudaMemset(dU, 0, sizeof(double) * m * n);

  // 对dA进行置零
  // launchKernel_ClearMatrix(gridDim, blockDim, m, n, dA, ldA);
  cudaMemset(dA, 0, sizeof(double) * m * n);


  int *com;
  cudaMalloc(&com, n*sizeof(int));

  // int* g_overFlag;
  // cudaMalloc(&g_overFlag, sizeof(int));
  // cudaMemset(g_overFlag, 0, sizeof(int));


  void *kernelArgs[] = {
      (void *)&n,
      (void *)&b,
      (void *)&dSubA,
      (void *)&ldSubA,
      (void *)&dU,
      (void *)&ldU,
      (void *)&blockNum,
      (void *)&com,
      // (void *)&g_overFlag,
  };
  dim3 dimBlock(64, 16, 1);
  dim3 dimGrid(blockNum, 1, 1);
  cudaLaunchCooperativeKernel((void *)chasing_kernel_one_timeV10<32>,
                              dimGrid,
                              dimBlock,
                              kernelArgs);
  // chasing_kernel_one_timeV10<32><<<dimGrid, dimBlock>>>(n,b,dSubA, ldSubA, dU,ldU, blockNum, com, g_overFlag);

  // 将条带化矩阵复制会原矩阵
  dim3 blockDimcpydSubA2dA(DIM_X);
  dim3 gridDimcpydSubA2dA((n + DIM_X-1) / DIM_X);

  kernel_bugle_chasing_cpydSubA2dA<<<gridDimcpydSubA2dA, blockDimcpydSubA2dA>>>(n,
                                                                                b,
                                                                                dSubA,
                                                                                ldSubA,
                                                                                dA,
                                                                                ldA);

  float g_bugle_chasing_kernel_time = stopTimer();

  CUDA_RT_CALL(cudaGetLastError());
  cudaDeviceSynchronize();

  printf("SY2TR dA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dA, ldA, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);

}
