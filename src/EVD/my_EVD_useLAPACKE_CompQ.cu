
#include <cstring>
#include <iomanip> // 包含操纵器的头文件
#include <iostream>
// #include <string>
// #include <vector>
// #include <lapacke.h>
// #include <mkl_lapacke.h>
#include <cusolverDn.h>
#include <algorithm> // std::sort

// #include "bugle_chasing.h"
// #include "bugle_chasingV3.h"
// #include "BC_backTrans.h"
#include "bugle_chasingV10.h"
#include "fileOpTool.h"
#include "kernelOther.h"
#include "myBase.h"
#include "zy_zy_sy2sb.h"

#include "computerWYFromSlide.h"
#include "computerQFromWY.h"
#include "checkMetric.h"

// #include <magma_v2.h>
// #include <magma_lapack.h>
// #include <magma_auxiliary.h>

// #define TESTING_CHECK( err )                                                 \
//     do {                                                                     \
//         magma_int_t err_ = (err);                                            \
//         if ( err_ != 0 ) {                                                   \
//             fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
//                      #err, __FILE__, __LINE__,                               \
//                      (long long) err_, magma_strerror(err_) );               \
//             exit(1);                                                         \
//         }                                                                    \
//     } while( 0 )



using namespace std;

float g_sy2sb_time                = 0.0;
float g_bugle_chasing_kernel_time = 0.0;

float g_chasing_kernel_computer_time = 0.0;

float g_cusolverSy2tr_Time = 0.0;

extern float g_panelQR_time_ZY;
extern float g_tc_ozimmu_syr2k_ZY;
extern float g_gemm_time_ZY;

#define CUSOLVER_CHECK 0
#define CHECK_BUGLE_CHASING_Q 1

#define BUGLE_CHASING_COMPUTER_Q 0

constexpr int DIM_X = 16;
constexpr int DIM_Y = 16;

// __device__ int com[128] = {0};

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
  cout << "n=" << n << ", b=" << b << ", nb=" << nb << endl;


  cusolverDnHandle_t cusolver_handle;
  cublasHandle_t cublas_handle;

  cusolverDnCreate(&cusolver_handle);
  cublasCreate(&cublas_handle);


  double *dT, *dA;
  CUDA_RT_CALL(cudaMalloc(&dT, sizeof(double) * m * n));
  CUDA_RT_CALL(cudaMalloc(&dA, sizeof(double) * m * n));

  // 使用cuda的方式生成随机矩阵
  generateUniformMatrix(dT, m, n);

  // 将dA转换为对称矩阵
  dim3 gridDim((m + DIM_X - 1) / DIM_X, (n + DIM_Y - 1 ) / DIM_Y);
  dim3 blockDim(DIM_X, DIM_Y);
  launchKernel_CpyMatrixL2U(gridDim, blockDim, n, dT, n);

  launchKernel_copyMatrix(gridDim, blockDim, m, n, dT, m, dA, m);

  // printAndWriteMatrixToCsvV2(dA, m, m, n, "Symatrix_A_128x128.csv"); 


  CUDA_RT_CALL(cudaFree(dT));

#define USE_MATRIX_FILE 1
#if USE_MATRIX_FILE
  string fileName = "/work/home/szhang94/wanghs/SugaSyEVD/data/Symatrix_A_128x128.csv";
  vector<vector<double>> data = readMatrixFromFile(fileName);

  m = data.size();
  n = data[0].size();
  double *A;

  A = (double *)malloc(sizeof(double) * m * n);

  fillMatrix(A, data);
  cudaMemcpy(dA, A, sizeof(double) * m * n, cudaMemcpyHostToDevice);
  free(A);

  // printf("Origin dA:\n");
  // // 打印开头的3x3个元A
  // printDeviceMatrixV2(dA, m, 3, 3);

  // // 打印结尾的3x3个元素
  // printDeviceMatrixV2(dA + (m - 3) + (n - 3) * m, m, 3, 3);
#endif

  double *dwork, *dR, *dW, *dY, *dZ;

  cudaMalloc(&dwork, sizeof(double) * (m + nb) * (n + nb));
  // cudaMalloc(&dR, sizeof(double) * m * n);
  cudaMalloc(&dW, sizeof(double) * m * n);
  cudaMalloc(&dY, sizeof(double) * m * n);
  // CUDA_RT_CALL(cudaMalloc(&dwork, sizeof(double) * m * nb));
  CUDA_RT_CALL(cudaMalloc(&dR, sizeof(double) * m * nb));
  // CUDA_RT_CALL(cudaMalloc(&dW, sizeof(double) * m * nb));
  // CUDA_RT_CALL(cudaMalloc(&dY, sizeof(double) * m * nb));

  CUDA_RT_CALL(cudaMalloc(&dZ, sizeof(double) * m * nb));

  int *info;
  CUDA_RT_CALL(cudaMalloc(&info, sizeof(int)));

  CHECK(cudaGetLastError());

  double *dOriA_1;
  cudaMalloc(&dOriA_1, sizeof(double) * m * n);


  launchKernel_copyMatrix(gridDim, blockDim, m, n, dA, m, dOriA_1, m);

  long ldOriA_1, ldA, ldW, ldY, ldZ, ldR;
  ldA      = m;
  ldOriA_1 = ldW = ldY = ldZ = ldR = m;

#define CHECK_SBR_ACCURARY 1
#if CHECK_SBR_ACCURARY
  launchKernel_ClearMatrix(gridDim, blockDim, m, n, dY, ldY);
  launchKernel_ClearMatrix(gridDim, blockDim, m, n, dW, ldW);

  double *dOriA, *dOriA_2;
  cudaMalloc(&dOriA, sizeof(double) * m * n);
  cudaMalloc(&dOriA_2, sizeof(double) * m * n);
  launchKernel_copyMatrix(gridDim, blockDim, m, n, dA, m, dOriA, m);
#endif  

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  printf("Begin dA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dA, ldA, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3); 


  startTimer();
  my_ZY_ZY_SBR_Vector(cusolver_handle,
               cublas_handle,
               m,
               n,
               b,
               nb,
               dOriA_1,
               ldOriA_1,
               dA,
               ldA,
               dW,
               ldW,
               dY,
               ldY,
               dZ,
               ldZ,
               dR,
               ldR,
               dwork,
               info);

  launchKernel_CpyMatrixL2U(gridDim, blockDim, m, dA, ldA);
  g_sy2sb_time = stopTimer();

  printf("SBR dA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dA, ldA, 3, 3);


  // 打印结尾的3x3个元素
  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);

  // 检查SBR的正确性
// #define CHECK_SBR_ACCURARY 1
#if CHECK_SBR_ACCURARY
    // 计算W和Y
  computerWYFromSlide(cublas_handle, m, n, b, dW, ldW, dY, ldY, dwork);

  // 计算Q
  computerQFromWY(cublas_handle, m, n, dOriA_1, ldA, dW, ldW, dY, ldY);

#if MY_DEBUG
  cout << "P and w dQ:" << endl;
  printAndWriteMatrixToCsvV2(dOriA_1, ldOriA_1, m, n, "WY_ZY_Q.csv");
#endif

  // 计算Q的正交性,
  // 1.复制Q到dOriA_2
  launchKernel_copyMatrix(gridDim, blockDim, m, n, dOriA_1, ldA, dOriA_2, ldA);
  // 计算正交性误差
  checkOrthogonality(cublas_handle, m, n, dOriA_1, ldA, dOriA_2, ldA, dwork);

  // 计算反向误差
  double done = 1.0;
  double dzero = 0.0;
  double dnegone = -1.0;

  long ldWork = ldA;

  // 1.复制OriA到dOriA_2
  launchKernel_copyMatrix(gridDim, blockDim, m, n, dOriA, ldA, dOriA_2, ldA);

  // ||oriA-Q*B*Q'||
  // 1.计算work=B*Q'
  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, m, &done, dA, ldA,
              dOriA_1, ldA, &dzero, dwork, ldWork);

  // 2.计算oriA=oriA-Q*work
  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &dnegone, dOriA_1, ldA,
              dwork, ldWork, &done, dOriA_2, ldA);

  // 计算(//A//)/(//oriA))//
  double snA, snOriA;
  int incx = 1;

  cublasDnrm2(cublas_handle, m * n, dOriA_2, incx, &snA);
  cublasDnrm2(cublas_handle, m * n, dOriA, incx, &snOriA);

  cout << "Backforward err: " << snA / snOriA / m << std::endl;

  printf("My ZY(acculumate) aglorithim size %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_sy2sb_time,
          2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_sy2sb_time * 1e9));
#endif  


#if 1  
  cudaFree(dR);
  cudaFree(dW);
  cudaFree(dY);
  cudaFree(dZ);

  cudaFree(dOriA_1);
  cudaFree(dwork);


  double *dSubA;
  cudaMalloc(&dSubA, sizeof(double) * (2 * b) * n);
  int ldSubA = 2 * b;

#if CHECK_BUGLE_CHASING_Q
  double *OA;
  long ldOA = m;
  cudaMalloc(&OA, sizeof(double) * m * n);
  launchKernel_copyMatrix(gridDim, blockDim, m, n, dA, m, OA, m);
#endif

  CHECK(cudaGetLastError());
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
  int numThreads = 32 * 32;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);


  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  //     &numBlocksPerSm,
  //     chasing_kernel_one_timeV10<32>,
  //     numThreads,
  //     0);
  // int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;


  // launch
  int blockNum = deviceProp.multiProcessorCount;

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

  launchKernel_ClearMatrix(gridDim, blockDim, m, n, dU, ldU);

  // 对dA进行置零
  launchKernel_ClearMatrix(gridDim, blockDim, m, n, dA, ldA);


  int *com;
  cudaMalloc(&com, n*sizeof(int));

  int* g_overFlag;
  cudaMalloc(&g_overFlag, sizeof(int));
  cudaMemset(g_overFlag, 0, sizeof(int));


  void *kernelArgs[] = {
      (void *)&n,
      (void *)&b,
      (void *)&dSubA,
      (void *)&ldSubA,
      (void *)&dU,
      (void *)&ldU,
      (void *)&blockNum,
      (void *)&com,
      (void *)&g_overFlag,
  };
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(blockNum, 1, 1);
  // cudaLaunchCooperativeKernel((void *)chasing_kernel_one_timeV10<32>,
  //                             dimGrid,
  //                             dimBlock,
  //                             kernelArgs);
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

  g_bugle_chasing_kernel_time = stopTimer();

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  printf("SY2TR dA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dA, ldA, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);


    
#if BUGLE_CHASING_COMPUTER_Q
  double *dQ;
  cudaMalloc(&dQ, sizeof(double) * m * n);
  long ldQ = m;

  launchKernel_ClearMatrix(gridDimClrDu, blockDimClrDU, m, n, dQ, ldQ);
  double value = 1.0;
  dim3 blockDimSetValue(DIM_X);
  dim3 gridDimSetValue((n + DIM_X-1) / DIM_X);
  launchKernel_setMetrixTrValue(gridDimSetValue, blockDimSetValue, n, n, dQ, ldQ, value);

  startTimer();

  // 调用函数求Q
  my_BC_back_trans_v8_10(dQ, n, dU, n, n, b);

  g_chasing_kernel_computer_time = stopTimer();

  printf("bugle chasing Q:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dQ, ldQ, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(dQ + (m - 3) + (n - 3) * ldQ, ldQ, 3, 3);

#if CHECK_BUGLE_CHASING_Q

  // 计算反向误差
  double done    = 1.0;
  double dzero   = 0.0;
  double dnegone = -1.0;

  double *oriA;
  cudaMalloc(&oriA, sizeof(double) * m * n);
  launchKernel_copyMatrix(gridDim, blockDim, m, n, OA, m, oriA, m);

  printf("bugle chasing OA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(OA, ldOA, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(OA + (m - 3) + (n - 3) * ldOA, ldOA, 3, 3);

  printf("bugle chasing oriA:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(oriA, m, 3, 3);

  // 打印结尾的3x3个元素
  printDeviceMatrixV2(oriA + (m - 3) + (n - 3) * m, m, 3, 3);

  // 1. 计算OA-Q'*dA*Q
  double *work;
  cudaMalloc(&work, sizeof(double) * m * n);
  long ldWork = m;

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // ||OA-Q'*dA*Q||
  // 1.计算work=dA*Q
  cublasDgemm(cublas_handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              m,
              n,
              m,
              &done,
              dA,
              ldA,
              dQ,
              ldQ,
              &dzero,
              work,
              ldWork);

  // 2.OA=OA-Q'*work
  cublasDgemm(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              m,
              n,
              m,
              &dnegone,
              dQ,
              ldQ,
              work,
              ldWork,
              &done,
              OA,
              ldOA);

  // 计算(//A//)/(//oriA))//
  double snA, snOriA;
  int incx = 1;

  cublasDnrm2(cublas_handle, m * n, OA, incx, &snA);
  cublasDnrm2(cublas_handle, m * n, oriA, incx, &snOriA);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  cout << "Backforward err: " << snA / snOriA / m << std::endl;

#endif
  
#endif

#define USE_LAPACKE 0
#if USE_LAPACKE

  double *mD, *mE;
  cudaMalloc(&mD, sizeof(double) *n);
  cudaMalloc(&mE, sizeof(double) *(n - 1));

  double *hS, *hE;
  hS  = (double *)malloc(sizeof(double) * n);
  hE  = (double *)malloc(sizeof(double) * (n - 1));

  startTimer();
  dim3 blockDim4(DIM_X);
  dim3 gridDim4((m + DIM_X-1) / DIM_X);
  launch_kernel_cpyATr2Vector(gridDim4, blockDim4, n, n, dA, ldA, mD);

  dim3 gridDim5((m - 1 + DIM_X-1) / DIM_X);
  launch_kernel_cpyATr2Vector(gridDim5, blockDim4, n - 1, n - 1, dA + 1, ldA, mE);

  cudaMemcpy(hS, mD, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hE, mE, sizeof(double) * (n - 1), cudaMemcpyDeviceToHost);

  // 为了节约内存
  cudaFree(dA);

  double *Z;
  Z = (double *)malloc(sizeof(double) * n*n);

  lapack_int lapckInfo=0;

  lapckInfo = LAPACKE_dstedc(LAPACK_COL_MAJOR, 'I', n, hS, hE, Z, n);

#define DEFINE_MAGMA_DC 0
#if DEFINE_MAGMA_DC
  magma_int_t il = 1;
  magma_int_t iu = n;
  double vl = 0.0, vu = 0.0;              // 不使用范围选择
  // magma_int_t m;                          // 实际找到的特征值数量
  magma_int_t *iwork;
  magma_int_t liwork = 3 + 5*n;
  double *work;
  magma_int_t lwork = 1 + 4 * n + n * n; //3287553; //用于1024
  magma_int_t magmaInfo;

  double *dwedc;
  TESTING_CHECK(magma_dmalloc_pinned(&dwedc, 3 * n * (n / 2 + 1)));

  // 分配工作空间
  // iwork = (magma_int_t*) malloc(liwork * sizeof(magma_int_t));
  // if(NULL == iwork)
  // {
  //   printf("malloc iwork fail.\n");
  // }
  TESTING_CHECK( magma_imalloc_cpu( &iwork, liwork ));

  // work = (double*) malloc(lwork * sizeof(double));
  TESTING_CHECK(magma_dmalloc_pinned(&work, lwork));
  // cudaMallocManaged(&work, sizeof(double) * lwork);

  double* Z = NULL;
  TESTING_CHECK(magma_dmalloc_pinned(&Z, n*n));
  // cudaMallocManaged(&Z, sizeof(double) * n * n);
  if(NULL == Z)
  {
    std::cout << "error" << std::endl;
  }
  // cudaMemcpy(Z, dA, sizeof(double) * n*n, cudaMemcpyDeviceToHost);

  double *hPE = NULL;
  TESTING_CHECK(magma_dmalloc_pinned(&hPE, n-1));
  if(NULL == hPE)
  {
    std::cout << "error" << std::endl;
  }
  // for(int k =0; k < n; k++)
  // {
  //   hPE[k] = 0.0;
  // }

  // hPE[3] = 1.0;
  memcpy(hPE, hE, (n-1)*sizeof(double));

  double *hPS= NULL;
  TESTING_CHECK(magma_dmalloc_pinned(&hPS, n));
  if(NULL == hPS)
  {
    std::cout << "error" << std::endl;
  }
  memcpy(hPS, hS, n*sizeof(double));

  // double *hPE= work;
  // memcpy(hPE, hE, n*sizeof(double));

  // magma_int_t sizE_onwork = n;

  // magma_int_t sizTAU1 = n;

  // magma_int_t sizTAU2 =4608;
  // magma_int_t sizV2 =884736;
  // magma_int_t sizT2 =294912;


  // double *TAU1 = work + sizE_onwork;
  // double *TAU2 = TAU1 + sizTAU1;
  // double *V2 = TAU2 + sizTAU2;
  // double *T2 = V2 + sizV2;
  // double *Wstg1 = T2 + sizT2;
  // // PAY ATTENTION THAT work[indA2] should be able to be of size lda2*n
  // // which it should be checked in any future modification of lwork.*/
  // double *A2 = Wstg1;
  // double *Z = Wstg1;

  // double *Wedc = Wstg1 + n * n;
  // magma_int_t lwedc =  1 + 4 * n + n * n;

  magma_int_t lapckInfo = magma_dstedx(MagmaRangeAll, n, vl, vu, il, iu, hPS, hPE, Z, n, work, lwork, iwork, liwork,dwedc, &magmaInfo);
#endif

  float lapackestedcTime = stopTimer();

  if(0 != lapckInfo)
  {
    std::cout <<"call LAPACKE_dstedc error" << std::endl;
  }

  std::vector<double> hDV(hS, hS + n); // 把数据从hD拷贝到vectoS
  std::sort(hDV.begin(), hDV.end());   // 对vector中的数据进行排序
  int count = 0;

  std::cout << std::fixed << std::setprecision(12);
  cout << "cusolver EVD Value:" << endl;
  for (const auto &val : hDV)
  {
      if ((count < 5) || (count >= (n - 5)))
      {
          std::cout << val << ' ';
      }

      count++;
  }
  std::cout << '\n';

  printf("magma dZ:\n");
  // 打印开头的3x3个元A
  printMatrix(Z, n, 3, 3);
  // printDeviceMatrixV2(dA, ldA, 3, 3);

  // 打印结尾的3x3个元素
  printMatrix(Z + (m - 3) + (n - 3) * n, n, 3, 3);
  // printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);

#endif

  printf("gemm %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_gemm_time_ZY,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_gemm_time_ZY * 1e9));

  printf("syr2k %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_tc_ozimmu_syr2k_ZY,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_tc_ozimmu_syr2k_ZY * 1e9));

  printf("qr %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_panelQR_time_ZY,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_panelQR_time_ZY * 1e9));

  printf("sy2sb %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_sy2sb_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_sy2sb_time * 1e9));

  printf("Bugle chasing %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_bugle_chasing_kernel_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_bugle_chasing_kernel_time * 1e9));

#if USE_LAPACKE
  printf("DC %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         lapackestedcTime,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (lapackestedcTime * 1e9));
#endif

  printf("Bugle chasing Compute Q %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_chasing_kernel_computer_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_chasing_kernel_computer_time * 1e9));

  float ms = g_sy2sb_time + g_bugle_chasing_kernel_time;
  printf("sy2tr %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         ms,
         (4.0 * n * n * n / 3.0) / (ms * 1e9));

#endif

  // cout << "printdQ: " << endl;
  // printDeviceMatrixV2(dA, m, m, n);

  // free(A);
}
