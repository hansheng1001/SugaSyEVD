

// #include <cuda_runtime.h>

// #include <cstdlib>

#include "fileOpTool.h"
#include "myBase.h"
// #include <__clang_cuda_runtime_wrapper.h>

static __inline__ __device__ double warpAllReduceSum(double val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

static __inline__ __device__ double warpAllReduceSumV2(double val, int ThreadCount = 32)
{
  for (int mask = ThreadCount / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

#define U_COUNT 8
#define U_LEN_PROC_1TIME (U_COUNT * 32)

// #define U_COL_COUNT 0

#define MAX_WARP_COUNT 24

#define SYNC_THREAD_NUM (32/U_COUNT)

#define U_COL_EXRTERN_COUNT 90

// #define U_COL_COUNT_TOTAL (U_COL_COUNT+U_COL_EXRTERN_COUNT)

// #define RQ_LEN 


// V5:每次读取64个u,每次处理1列中的4个u
// V6:在V5的基础上,取消预取
extern __shared__ double externSM[];
__global__ void BC_kernel_computerQ_1Col_V8_10(int n,
                                            int perBlockN,
                                            int sweepCount,
                                            int lastSweepUCount,
                                            double *dCU,
                                            // long countU,
                                            double *dQ,
                                            long ldQ)
{

  // 也用于存放u
  double* sU2 = externSM;

  __shared__ double stailQ[MAX_WARP_COUNT*U_COL_EXRTERN_COUNT];

  __shared__ double stailQW[MAX_WARP_COUNT*U_COL_EXRTERN_COUNT];


  __shared__ double sTData[MAX_WARP_COUNT*32];

  double rQ[U_COUNT]; //使用寄存器替换shared memory来存储数据

  // double rU[U_COUNT];
  double4 *rQ4 = (double4*)rQ;

  int bInx = blockIdx.x;

  int i = threadIdx.x;
  int j = threadIdx.y;

  dQ = dQ + bInx*perBlockN*ldQ;

  if(bInx == gridDim.x-1)
  {
    perBlockN =  n - bInx*perBlockN;
  }

  int sweepIndex;

  int totalU; // 每次处理1列中的8个u

  long sweepBaseRow;
  long indexU = 0;

  #pragma unroll
  for (sweepIndex = 0; sweepIndex < sweepCount; sweepIndex++)
  {
    sweepBaseRow = (sweepCount - sweepIndex - 1) * U_LEN_PROC_1TIME;
    // totalU = (U_LEN_PROC_1TIME - 2) + sweepIndex * U_LEN_PROC_1TIME; 

    totalU = lastSweepUCount + sweepIndex * U_LEN_PROC_1TIME; 

    #pragma unroll
    for (;totalU > 0;)
    {
      __syncthreads();

      // int procUCOl = min(U_COL_EXRTERN_COUNT, totalU);

      // 读取动态内存中可以存放的u
      #pragma unroll
      for(int k =j; k<U_COL_EXRTERN_COUNT;k += MAX_WARP_COUNT)
      {
        #pragma unroll
        for(int t =0; t<U_COUNT;t++)
        {
          sU2[k*U_LEN_PROC_1TIME+i + t *32] = dCU[(indexU+k)*U_LEN_PROC_1TIME + i*U_COUNT +t];
          // sU2[k*U_LEN_PROC_1TIME+i + t *32] = dCU[(indexU+k)*U_LEN_PROC_1TIME + i +t*32];
        }
      }

      __syncthreads();

      // #pragma unroll
      for (int k = j; k < perBlockN; k += MAX_WARP_COUNT)
      {
        // 计算Q
        // 3.2 计算u'*q
        // #pragma unroll
        // for (int t = 0; t < U_COUNT; t++)
        // {
        //   rQ[t] = dQ[k * ldQ + sweepBaseRow + i*U_COUNT + t];
        // }

        double4 *tmpDQ4 = (double4*)(dQ+k * ldQ + sweepBaseRow);
        #pragma unroll
        for (int t = 0; t < U_COUNT/4; t++)
        {
          rQ4[t] = tmpDQ4[i*U_COUNT/4 + t];
        }

        __syncwarp();

        // 读取每行尾部多余的需要参与运算的q
        #pragma unroll
        for (int t = i; t < U_COL_EXRTERN_COUNT; t +=32)
        {
          stailQ[j*U_COL_EXRTERN_COUNT + t] = dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + t];
        }
        
        // #pragma unroll
        // for (int t = 0; t < U_COL_EXRTERN_COUNT/32; t++)
        // {
        //   stailQ[j*U_COL_EXRTERN_COUNT + t*32 + i] = dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + t*32+i];
        // }

        // double3 * tmpDQ3= (double3*)(dQ+k * ldQ + sweepBaseRow+U_LEN_PROC_1TIME);
        // double3 * stailQ3 = (double3*)stailQ;
        // stailQ3[j*U_COL_EXRTERN_COUNT/3 + i] = tmpDQ3[i];

        __syncwarp();

        int h = 0;
        // 处理动态内存中的u
        #pragma unroll
        for (; h < U_COL_EXRTERN_COUNT; h++)
        {
          // 写入最上面的Q
          if (0 != i)
          {
            sTData[j*32 + i] = rQ[0];
          }else
          {
            // dQ[k * ldQ + sweepBaseRow + h] = rQ[0];

            // 将需要写入的元素先写入到共享内存中
            stailQW[j*U_COL_EXRTERN_COUNT+h] = rQ[0];
          }

          __syncwarp();

          // 进行数据的搬移
          #pragma unroll
          for (int t = 0; t < U_COUNT-1; t++)
          {
            rQ[t] = rQ[t+1];
          }

          if(31 != i)
          {
            // sQ[j * U_LEN_PROC_1TIME + i + 7 * 32] = sQ[j * U_LEN_PROC_1TIME + i + 1];
            rQ[U_COUNT-1] = sTData[j*32 + i+1];
          }else{
            // 最后1个线程读取数据
            // rQ[U_COUNT-1] = dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + h];
            rQ[U_COUNT-1] = stailQ[j*U_COL_EXRTERN_COUNT+h];
          }
          __syncwarp();

          double nux = 0.0;

          // #pragma unroll
          // for (int t = 0; t < U_COUNT; t++)
          // {
          //   rU[t] = sU2[h*U_LEN_PROC_1TIME+i + t * 32];
          //   // rU[t] = sU2[h*U_LEN_PROC_1TIME+i*U_COUNT + t];
          // }


          #pragma unroll
          for (int t = 0; t < U_COUNT; t++)
          {
            nux += sU2[h*U_LEN_PROC_1TIME+i + t * 32] * rQ[t];
            // nux += rU[t] * rQ[t];
          }

          nux = warpAllReduceSumV2(nux, SYNC_THREAD_NUM);

          #pragma unroll
          for (int t = 0; t < U_COUNT; t++)
          {
            rQ[t] -= nux * sU2[h*U_LEN_PROC_1TIME+i+t * 32];
            // rQ[t] -= nux * rU[t];
          }

        }

        // 写回Q
        // #pragma unroll
        // for (int t = 0; t < U_COUNT; t++)
        // {
        //   dQ[k * ldQ + sweepBaseRow + h + i*U_COUNT + t] = rQ[t];
        // }

        // 将缓存到共享内存中的Q写入到全局内存中
        #pragma unroll
        for (int t = i; t < U_COL_EXRTERN_COUNT; t +=32)
        {
          // stailQ[j*U_COL_EXRTERN_COUNT + t] = dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + t];
          dQ[k * ldQ + sweepBaseRow + t] = stailQW[j*U_COL_EXRTERN_COUNT + t];
        }

        tmpDQ4 = (double4*)(dQ+k * ldQ + sweepBaseRow + h);
        #pragma unroll
        for (int t = 0; t < U_COUNT/4; t++)
        {
          tmpDQ4[i*U_COUNT/4 + t] = rQ4[t];
        }
      }

      indexU += U_COL_EXRTERN_COUNT;
      totalU -= U_COL_EXRTERN_COUNT;

      sweepBaseRow += U_COL_EXRTERN_COUNT;

      __syncthreads();
    }


  }
}

int my_BC_back_trans_v8_10(double *Q, long ldQ, double *U, long ldU, long n, int b)
{
  // 1.先对U进行连续条带化存储--每个u都是32的长度,不够长度的进行补0。同时存放每个u的起始位置
  // 1.1 计算u的个数

  int sweepCount = (n - 1 - 1 + (U_LEN_PROC_1TIME - 1)) / (U_LEN_PROC_1TIME);

  std::cout << "sweepCount: " << sweepCount << std::endl;


  // 计算最后1趟有多少个u
  // 最后1趟的起始位置
  int lastSweepUCount = n - ((sweepCount-1)*U_LEN_PROC_1TIME+1)-1;

  long countU = 0;
  for (int i = 0, tmp = lastSweepUCount; i <sweepCount; i++, tmp+=U_LEN_PROC_1TIME)
  {
    int tmp2 = (tmp+U_COL_EXRTERN_COUNT-1)/U_COL_EXRTERN_COUNT*U_COL_EXRTERN_COUNT;
    countU += tmp2;
  }

  printf("countU: %ld\n", countU);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

#if 0
  printf("bugle chasing Q:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(Q, ldQ, n, n);

  // 打印结尾的3x3个元素
  // printDeviceMatrixV2(Q + n-3 + (n - 3) * ldQ, ldQ, 3, 3);

  printf("bugle chasing U:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(U, ldU, n, n);
#endif

  // 1.2 申请内存
  double *dCU;
  auto err1 = cudaMalloc(&dCU, sizeof(double) * countU * U_LEN_PROC_1TIME);
  if (err1 != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err1) << std::endl;
    return -1;
  }
  // malloc(sizeof(double) * count * b);

  double *dCU_base = dCU;

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

#if 1
  // 1.3 连续存储
  for (int i = 0; i < sweepCount; i++)
  {
    // 1.3.2 存储u的起始位置
    int base   = (sweepCount - i - 1) * U_LEN_PROC_1TIME + 1;
    int uCount = i * U_LEN_PROC_1TIME + lastSweepUCount;

    for (int j = 0; j < uCount; j++)
    {
      // 1.3.1 计算u的长度
      int len = std::min(U_LEN_PROC_1TIME, int(n - base)); // n-1-base+1

      // 1.3.2 存储u的数据
      auto err =
        cudaMemcpy(dCU_base, U + base + j * ldU, len * sizeof(double), cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess)
      {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
      }
      dCU_base += U_LEN_PROC_1TIME;

      CHECK(cudaGetLastError());
      cudaDeviceSynchronize();

      // 下1列的u的起始位置
      base += 1;
    }

    // 填充到U_COL_EXRTERN_COUNT的整数个u
    int tmp2 = (uCount+U_COL_EXRTERN_COUNT-1)/U_COL_EXRTERN_COUNT*U_COL_EXRTERN_COUNT;
    dCU_base += (tmp2-uCount)*U_LEN_PROC_1TIME;
  }
#endif

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // 2.将Q每列的最下面扩充b个0
  double *dEQ;
  cudaMalloc(&dEQ, sizeof(double) * (n + U_LEN_PROC_1TIME) * n);

  long ldEQ = n + U_LEN_PROC_1TIME;

  for (int i = 0; i < n; i++)
  {
    cudaMemcpy(dEQ + i * ldEQ, Q + i * ldQ, sizeof(double) * n, cudaMemcpyDeviceToDevice);
  }

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

#if 0
  printf("bugle chasing dEQ:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dEQ, ldEQ, ldEQ, n);


  printf("bugle chasing dCU:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dCU, U_LEN_PROC_1TIME, U_LEN_PROC_1TIME, 10);
#endif

  // 3.计算Q

  // ssize_t shareDyMem = 1024; // 动态共享内存大小
  // ssize_t shareDyMem = U_COL_COUNT * U_LEN_PROC_1TIME *8; // 动态共享内存大小
  ssize_t shareDyMem = U_COL_EXRTERN_COUNT* U_LEN_PROC_1TIME *8; // 动态共享内存大小
  cudaFuncSetAttribute(BC_kernel_computerQ_1Col_V8_10,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shareDyMem);

  // ssize_t shareDyMem = 0;

  // int carveout = 100;
  // cudaFuncSetAttribute(BC_kernel_computerQ_1Col_V8_8,
  //                      cudaFuncAttributePreferredSharedMemoryCarveout,
  //                      carveout);

  dim3 dimBlock(32, MAX_WARP_COUNT, 1);
#if 1
  int dev                = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

  printf("Device %d supports cooperative launch: %s\n", dev, supportsCoopLaunch ? "true" : "false");
  int numBlocksPerSm = 0;
  // Number of threads my_kernel will be launched with
  int numThreads = 32 * MAX_WARP_COUNT;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                                  BC_kernel_computerQ_1Col_V8_10,
                                                                  numThreads,
                                                                  shareDyMem);
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // launch
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;

  printf("sharedMemPerBlock: %d\n", (int)deviceProp.sharedMemPerBlock);

  printf("maxThreadsPerBlock: %d\n", deviceProp.maxThreadsPerBlock);
  printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
  printf("numBlocksPerSm: %d\n", numBlocksPerSm);

  // printf("numBlocksPerSm: %d\n", numBlocksPerSm);
  printf("blockNum: %d\n", blockNum);

  int perBlockN = (n+blockNum-1)/blockNum;

  // size_t sharedMem = 128 * 1024;

  // lastSweepCount = (lastSweepCount+U_COL_EXRTERN_COUNT-1)/U_COL_EXRTERN_COUNT*U_COL_EXRTERN_COUNT;

  printf("[1]: n = %d, b = %d, countU = %d, sweepCount = %d, blockCount =%d, lastSweepCount = %d.\n",
         n,
         b,
         countU,
         sweepCount,
         blockNum,
         lastSweepUCount);

  startTimer();
  void *kernelArgs[] = {(void *)&n,
                        (void *)&perBlockN,
                        (void *)&sweepCount,
                        (void *)&lastSweepUCount,
                        (void *)&dCU,
                        (void *)&dEQ,
                        (void *)&ldEQ};

  dim3 dimGrid(blockNum, 1, 1);
  cudaLaunchCooperativeKernel((void *)BC_kernel_computerQ_1Col_V8_10,
                              dimGrid,
                              dimBlock,
                              kernelArgs,
                              shareDyMem);
#else

  // 启动内核，指定动态共享内存大小
  BC_kernel_computerQ_1Col_V8<<<114, dimBlock>>>(n, b, dCU, countU, dEQ, ldEQ, sweepCount, 114);
#endif

  float g_SVD_BC_BackTans_Time = stopTimer();

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // 把dEQ拷贝回Q
  // 可以使用核函数进行优化
  for (int i = 0; i < n; i++)
  {
    cudaMemcpy(Q + i * ldQ, dEQ + i * ldEQ, sizeof(double) * ldQ, cudaMemcpyDeviceToDevice);
  }

#if 0
  printf("bugle chasing end dEQ:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(dEQ, ldEQ, ldEQ, n);

  printf("bugle chasing end Q:\n");
  // 打印开头的3x3个元A
  printDeviceMatrixV2(Q, ldQ, n, n);
#endif

  // 释放内存
  cudaFree(dCU);
  cudaFree(dEQ);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  printf("g_SVD_BC_Time: %f\n", g_SVD_BC_BackTans_Time);

  return 0;
}

#if 0
int main(int argc, char **argv)
{
  long n = 1024;
  int b  = 32;

  if (3 != argc)
  {
    std::cout << "Uage: App n b" << std::endl;
    return 0;
  }

  n = atol(argv[1]);
  b = atoi(argv[2]);

  printf("n: %ld, b: %d\n", n, b);

  cudaSetDevice(0);
  // cudaInitDevice();

  double *Q;
  cudaMalloc(&Q, sizeof(double) * n * n);

  double *U;
  cudaMalloc(&U, sizeof(double) * n * n);

  generateUniformMatrix(Q, n, n);

  generateUniformMatrix(U, n, n);

  my_BC_back_trans_v8(Q, n, U, n, n, b);
}
#endif