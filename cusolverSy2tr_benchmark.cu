
#include <string>
#include <vector>

#include <iostream>
#include <cusolverDn.h>
#include <curand.h>

// #include "kernelOther.h"
// #include "fileOpTool.h"
// #include "myBase.h"

#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// 下面的部分只在单个文件的内部进行调用
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

// template <>
static void generateUniformMatrix(double *dA, long int m, long int n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniformDouble(gen, dA, long(m * n));
}


static __global__ void copyMatrixL2U(long n, double *A, long ldA)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // printf("come %d\n", __LINE__);
  // __syncthreads();

  if (i < n && j < n)
  {
    if (j > i)
      A[i + j * ldA] = A[j + i * ldA];
  }
}


static void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, double *A, long ldA)
{
  copyMatrixL2U<<<gridDim, blockDim>>>(n, A, ldA);
}

using namespace std;

float g_cusolverSy2tr_Time = 0.0;
float g_cusolverSy2tr_calcQ_Time = 0.0;

int main(int argc, char *argv[])
{
    // 要求nb必须小于n，且n必须是nb的倍数
    long m, n;

    // string fileName = "/home/wanghs/hpc/QRFunc/data/Matrix_128x32.csv";
    // string fileName = "/home/wanghs/hpc/QRFunc/data/Matrix_128x128.csv";
    // string fileName = "/home/wanghs/hpc/QRFunc/data/ZY_B.csv";
    // string fileName = "/home/wanghs/hpc/QRFunc/data/ZY_B_8192x8192.csv";
    // string fileName = "/home/wanghs/hpc/QRFunc/data/Matrix_SBR_15x15_3.csv";
    if (argc >= 2)
    {
        // 获得文件名
        // fileName = argv[1];
    }

    m = n = atol(argv[1]);

    int testCase = atoi(argv[2]);

    // // 读取输入csv文件
    // vector<vector<double>> data = readMatrixFromFile(fileName);

    // m = data.size();
    // n = data[0].size();

    // // 将读取的数据填充到A中
    // double *A;
    // A = (double *)malloc(sizeof(double) * m * n);

    // fillMatrix(A, data);
    // cout << "printA: " << endl;
    // printMatrix(A, m, n);

    // 创建cusolver
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    double *dA, *dD, *dE, *tau, *dWork;
    cudaMalloc(&dA, sizeof(double) * m * n);
    cudaMalloc(&dD, sizeof(double) * n);
    cudaMalloc(&dE, sizeof(double) * (n - 1));
    cudaMalloc(&tau, sizeof(double) * (n - 1));

    // 1. 复制A中到dA中
    // cudaMemcpy(dA, A, sizeof(double) * m * n, cudaMemcpyHostToDevice);

    // 使用cuda的方式生成随机矩阵
    generateUniformMatrix(dA, m, n);

    // 将dA转换为对称矩阵
    dim3 gridDim1((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim1(32, 32);
    launchKernel_CpyMatrixL2U(gridDim1, blockDim1, n, dA, n);

    CHECK(cudaGetLastError());

    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // 求出dWork所需空间大小
    int lwork;
    cusolverDnDsytrd_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n,
                                dA, n, dD, dE, tau, &lwork);

    // 分配dwork空间
    cudaMalloc(&dWork, sizeof(double) * lwork);

    // my_hou_kernel<128, 32><<<gridDim, blockDim>>>(96, 32, dA + 32, m, dR, n);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    startTimer();
    cusolverDnDsytrd(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                     n, dA, n, dD, dE, tau, dWork, lwork, devInfo);
    g_cusolverSy2tr_Time = stopTimer();

 

    // cudaMemcpy(A, dA, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

    // cout << "P and W Trig: " << endl;
    // fileName = "cosovlersy2sb_Trig_" + to_string(m) + "x" + to_string(n) + ".csv";
    // printAndWriteMatrixToCsvV2(dA, m, m, n, fileName);

    // 打印开头的3x3个元素
    // printDeviceMatrixV2(dA, m, 3, 3);

    // // 打印结尾的3x3个元素
    // printDeviceMatrixV2(dA + (m - 3) + (n - 3) * m, m, 3, 3);

    printf("Benchmark cusolverSy2tr Sy2tr %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_cusolverSy2tr_Time,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_cusolverSy2tr_Time * 1e9));


    float ms = g_cusolverSy2tr_Time + g_cusolverSy2tr_calcQ_Time;
    printf("Benchmark cusolverSy2tr %ldx%ld takes %lf ms, tflops is %lf\n", m, n, ms,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (ms * 1e9));

    cudaFree(dWork);
    cudaFree(tau);
    cudaFree(dE);
    cudaFree(dD);
    cudaFree(dA);

    // free(A);
}
