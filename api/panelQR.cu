#include <string>
#include <vector>

#include <curand.h>
#include <cusolverDn.h>

// #include "myBase.h"
#include "fileOpTool.h"
#include "kernelOther.h"
#include "kernelQR.h"
#include "TallShinnyQR.h"

using namespace std;

#define MY_DEBUG 0

float g_QR_Time          = 0.0;
float g_Litter_GEMM_Time = 0.0;

// 此函数分解过后，A中存放的是Y矩阵，W中存放的是W矩阵，R中存放的是R矩阵
// #include "panelQR.h"

template <typename T>
void panelQR(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             T *A,
             long lda,
             T *W,
             long ldw,
             T *R,
             long ldr,
             T *work,
             int *info);

template <>
void panelQR(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             double *A,
             long lda,
             double *W,
             long ldw,
             double *R,
             long ldr,
             double *work,
             int *info)
{


  if (n <= 32)
  {
    startTimer();
#if MY_DEBUG
    cout << "print dA1:" << std::endl;
    string fileName = "dA1_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif
    // 1. 直接使用瘦高矩阵的QR分解方法进行QR分解
    // 使用此函数进行QR分解过后A中存放的是Q矩阵，R矩阵中存放的R矩阵
    // hou_tsqr_panel<double, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);
    hou_tsqr_panel<128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // 2.求出Y，因为Q=I-WY',而W=Y*T,所以Q=I-Y*T*Y',所以I-Q=Y*T*Y',显然Y是
    // 是下三角矩阵，对I-Q进行LU分解就可以得到Y了

    cout << "print dQ:" << std::endl;
    fileName = "dQ_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);

    // cout << "print dR:" << std::endl;
    // printDeviceMatrix(R, n, n);
    fileName = "dR_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(R, ldr, n, n, fileName);
#endif

    // 2.1 求I-Q
    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    // 通过这个函数以后，A中存放的是I-Q
    launchKernel_IminusQ(gridDim, blockDim, m, n, A, lda);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    cout << "print I-Q:" << std::endl;
    printDeviceMatrixV2(A, lda, m, n);
#endif

    // 复制A到W中
    launchKernel_copyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print W(I-Q):" << std::endl;
    printDeviceMatrixV2(W, ldw, m, n);
#endif

    // // cusolverStatus_t
    // int Lwork;
    // double *devWork = NULL; // workspace
    // cusolverDnDgetrf_bufferSize(cusolver_handle, m, n, A, lda, &Lwork);

    // cudaMalloc((void **)&devWork, sizeof(double) * Lwork);

    // // 2.2 对I-Q进行LU分解，得到Y
    // cusolverDnDgetrf(cusolver_handle, m, n, A, lda, devWork, NULL, info);
    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    // cudaFree(devWork);

    // 2.2 对I-Q进行LU分解，得到Y
    cusolverDnDgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);
    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    // 2.3 获取的A的下三角矩阵，得到Y
    launchKernel_getLower(gridDim, blockDim, m, n, A, lda);
    // launchKernel_ClearMatrix(gridDim, blockDim, m, n, W, lda);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print Y:" << std::endl;
    // printDeviceMatrixV2(A, lda, m, n);
    fileName = "dy_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif

    double done = 1.0;
    // 2.4 求出W,因为WY'=I-Q
    cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &done,
                A,
                lda,
                W,
                ldw);

    //     cublasStrsm(cublas_handle,
    //                 CUBLAS_SIDE_RIGHT,
    //                 CUBLAS_FILL_MODE_LOWER,
    //                 CUBLAS_OP_T,
    //                 CUBLAS_DIAG_NON_UNIT,
    //                 m,
    //                 n,
    //                 &done,
    //                 A,
    //                 lda,
    //                 W,
    //                 ldw);

    //     cublasDtrsm(cublas_handle,
    //                 CUBLAS_SIDE_RIGHT,
    //                 CUBLAS_FILL_MODE_LOWER,
    //                 CUBLAS_OP_T,
    //                 CUBLAS_DIAG_NON_UNIT,
    //                 m,
    //                 n,
    //                 &done,
    //                 A,
    //                 lda,
    //                 W,
    //                 ldw);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print W:" << std::endl;
    fileName = "dw_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(W, ldw, m, n, fileName);
#endif
    g_QR_Time += stopTimer();

    return;
  }

  // 1. 对A的前一半进行递归调用panelQR，得到W、Y和R
  panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);

  // 2. 计算A的后一半，也就是A2=A2-YW'A2
    double done = 1.0, dzero = 0.0, dnegone = -1.0;
//   double tone    = 1.0;
//   double tzero   = 0.0;
//   double tnegone = -1.0;

  startTimer();
  // 2.1 先计算W'A2,把结果存放到work中

//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_T,
//                CUBLAS_OP_N,
//                n / 2,
//                n - n / 2,
//                m,
//                &tone,
//                W,
//                cuda_data_type,
//                ldw,

//                A + n / 2 * lda,
//                cuda_data_type,
//                lda,

//                &tzero,
//                work,
//                cuda_data_type,
//                n / 2,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);

    cublasDgemm(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n / 2,
                n - n / 2,
                m,
                &done,
                W,
                ldw,
                A + n / 2 * lda,
                lda,
                &dzero,
                work,
                n / 2);

  // 2.2 计算A2-YW'A2,也就是计算A2-Y*work
//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_N,
//                CUBLAS_OP_N,
//                m,
//                n - n / 2,
//                n / 2,
//                &tnegone,
//                A,
//                cuda_data_type,
//                lda,

//                work,
//                cuda_data_type,
//                n / 2,

//                &tone,
//                A + n / 2 * lda,
//                cuda_data_type,
//                lda,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);

    cublasDgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n - n / 2,
                n / 2,
                &dnegone,
                A,
                lda,
                work,
                n / 2,
                &done,
                A + n / 2 * lda,
                lda);
  g_Litter_GEMM_Time += stopTimer();

#if MY_DEBUG
  cout << "print dA2:" << std::endl;
  string fileName = "dA2_" + to_string(m) + "_" + to_string(n) + ".csv";
  printAndWriteMatrixToCsvV2(A + n / 2 * lda, lda, m, n / 2, fileName);
#endif

  // 2.3 将A的右上角部分拷贝到R中，同时对A的右上角部分清零
  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchKernel_copyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);

  // 3. 对A的后半部分递归调用panelQR，得到W、Y和R
  panelQR(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);

  // 4. 求出新的W和Y，注意因为Y=[y1|y2],所以Y直接就已经求出来了，所以只用求W。
  // 因为W=[w1|w2-w1y1'w2],w1部分已经是求好的了，所以只需要求w2-w1y1'w2
  // 4.1 求y1'w2,把结果放到work中
  // 由于w2的上半部分为0,所以进行运算时可以从A+n/2开始
  startTimer();
//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_T,
//                CUBLAS_OP_N,
//                n / 2,
//                n - n / 2,
//                m - n / 2,
//                &tone,
//                A + n / 2,
//                cuda_data_type,
//                lda,

//                W + n / 2 + n / 2 * ldw,
//                cuda_data_type,
//                ldw,

//                &tzero,
//                work,
//                cuda_data_type,
//                n / 2,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);
    cublasDgemm(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n / 2,
                n - n / 2,
                m - n / 2,
                &done,
                A + n / 2,
                lda,
                W + n / 2 + n / 2 * ldw,
                ldw,
                &dzero,
                work,
                n / 2);

  // 4.2 求w2-w1y1'w2, 也即是w2-w1*work
//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_N,
//                CUBLAS_OP_N,
//                m - n / 2,
//                n - n / 2,
//                n / 2,
//                &tnegone,
//                A + n / 2,
//                cuda_data_type,
//                lda,

//                work,
//                cuda_data_type,
//                n / 2,

//                &tone,
//                W + n / 2,
//                cuda_data_type,
//                ldw,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);
    cublasDgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n - n / 2,
                n / 2,
                &dnegone,
                W,
                ldw,
                work,
                n / 2,
                &done,
                W + n / 2 * ldw,
                ldw);
  g_Litter_GEMM_Time += stopTimer();

  return;
}

// template <typename T>
// void panelQR(cusolverDnHandle_t cusolver_handle,
//              cublasHandle_t cublas_handle,
//              long m,
//              long n,
//              T *A,
//              long lda,
//              T *W,
//              long ldw,
//              T *R,
//              long ldr,
//              T *work,
//              int *info);

// 实例化模版函数
// template void panelQR<double>(cusolverDnHandle_t,
//                               cublasHandle_t,
//                               long,
//                               long,
//                               double *,
//                               long,
//                               double *,
//                               long,
//                               double *,
//                               long,
//                               double *,
//                               int *);

template <>
void panelQR(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             float *A,
             long lda,
             float *W,
             long ldw,
             float *R,
             long ldr,
             float *work,
             int *info)
{

  if (n <= 32)
  {
    startTimer();
#if MY_DEBUG
    cout << "print dA1:" << std::endl;
    string fileName = "dA1_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif
    // 1. 直接使用瘦高矩阵的QR分解方法进行QR分解
    // 使用此函数进行QR分解过后A中存放的是Q矩阵，R矩阵中存放的R矩阵
    // hou_tsqr_panel<float, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);
    hou_tsqr_panel<128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // 2.求出Y，因为Q=I-WY',而W=Y*T,所以Q=I-Y*T*Y',所以I-Q=Y*T*Y',显然Y是
    // 是下三角矩阵，对I-Q进行LU分解就可以得到Y了

    cout << "print dQ:" << std::endl;
    fileName = "dQ_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);

    // cout << "print dR:" << std::endl;
    // printDeviceMatrix(R, n, n);
    fileName = "dR_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(R, ldr, n, n, fileName);
#endif

    // 2.1 求I-Q
    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    // 通过这个函数以后，A中存放的是I-Q
    launchKernel_IminusQ(gridDim, blockDim, m, n, A, lda);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    cout << "print I-Q:" << std::endl;
    printDeviceMatrixV2(A, lda, m, n);
#endif

    // 复制A到W中
    launchKernel_copyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print W(I-Q):" << std::endl;
    printDeviceMatrixV2(W, ldw, m, n);
#endif

    // // cusolverStatus_t
    // int Lwork;
    // double *devWork = NULL; // workspace
    // cusolverDnDgetrf_bufferSize(cusolver_handle, m, n, A, lda, &Lwork);

    // cudaMalloc((void **)&devWork, sizeof(double) * Lwork);

    // // 2.2 对I-Q进行LU分解，得到Y
    // cusolverDnDgetrf(cusolver_handle, m, n, A, lda, devWork, NULL, info);
    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    // cudaFree(devWork);

    // 2.2 对I-Q进行LU分解，得到Y
    cusolverDnSgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);

    // 并未去使用cusolverDnXgetrf--这个函数可以使用
    // cusolverDnXgetrf(cusolver_handle,NULL, m, n, cuda_data_type, A, lda, work, info);

    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    // 2.3 获取的A的下三角矩阵，得到Y
    launchKernel_getLower(gridDim, blockDim, m, n, A, lda);
    // launchKernel_ClearMatrix(gridDim, blockDim, m, n, W, lda);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print Y:" << std::endl;
    // printDeviceMatrixV2(A, lda, m, n);
    fileName = "dy_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif

    float fone = 1.0;
    // 2.4 求出W,因为WY'=I-Q
    cublasStrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &fone,
                A,
                lda,
                W,
                ldw);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print W:" << std::endl;
    fileName = "dw_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(W, ldw, m, n, fileName);
#endif
    g_QR_Time += stopTimer();

    return;
  }

  // 1. 对A的前一半进行递归调用panelQR，得到W、Y和R
  panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);

  // 2. 计算A的后一半，也就是A2=A2-YW'A2
  //   double done = 1.0, dzero = 0.0, dnegone = -1.0;
  float tone    = 1.0;
  float tzero   = 0.0;
  float tnegone = -1.0;

  startTimer();
  // 2.1 先计算W'A2,把结果存放到work中

//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_T,
//                CUBLAS_OP_N,
//                n / 2,
//                n - n / 2,
//                m,
//                &tone,
//                W,
//                cuda_data_type,
//                ldw,

//                A + n / 2 * lda,
//                cuda_data_type,
//                lda,

//                &tzero,
//                work,
//                cuda_data_type,
//                n / 2,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);

    cublasSgemm(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n / 2,
                n - n / 2,
                m,
                &tone,
                W,
                ldw,
                A + n / 2 * lda,
                lda,
                &tzero,
                work,
                n / 2);

  // 2.2 计算A2-YW'A2,也就是计算A2-Y*work
//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_N,
//                CUBLAS_OP_N,
//                m,
//                n - n / 2,
//                n / 2,
//                &tnegone,
//                A,
//                cuda_data_type,
//                lda,

//                work,
//                cuda_data_type,
//                n / 2,

//                &tone,
//                A + n / 2 * lda,
//                cuda_data_type,
//                lda,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);

    cublasSgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n - n / 2,
                n / 2,
                &tnegone,
                A,
                lda,
                work,
                n / 2,
                &tone,
                A + n / 2 * lda,
                lda);
  g_Litter_GEMM_Time += stopTimer();

#if MY_DEBUG
  cout << "print dA2:" << std::endl;
  string fileName = "dA2_" + to_string(m) + "_" + to_string(n) + ".csv";
  printAndWriteMatrixToCsvV2(A + n / 2 * lda, lda, m, n / 2, fileName);
#endif

  // 2.3 将A的右上角部分拷贝到R中，同时对A的右上角部分清零
  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchKernel_copyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);

  // 3. 对A的后半部分递归调用panelQR，得到W、Y和R
  panelQR(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);

  // 4. 求出新的W和Y，注意因为Y=[y1|y2],所以Y直接就已经求出来了，所以只用求W。
  // 因为W=[w1|w2-w1y1'w2],w1部分已经是求好的了，所以只需要求w2-w1y1'w2
  // 4.1 求y1'w2,把结果放到work中
  // 由于w2的上半部分为0,所以进行运算时可以从A+n/2开始
  startTimer();
//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_T,
//                CUBLAS_OP_N,
//                n / 2,
//                n - n / 2,
//                m - n / 2,
//                &tone,
//                A + n / 2,
//                cuda_data_type,
//                lda,

//                W + n / 2 + n / 2 * ldw,
//                cuda_data_type,
//                ldw,

//                &tzero,
//                work,
//                cuda_data_type,
//                n / 2,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);
    cublasSgemm(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n / 2,
                n - n / 2,
                m - n / 2,
                &tone,
                A + n / 2,
                lda,
                W + n / 2 + n / 2 * ldw,
                ldw,
                &tzero,
                work,
                n / 2);

  // 4.2 求w2-w1y1'w2, 也即是w2-w1*work
//   cublasGemmEx(cublas_handle,
//                CUBLAS_OP_N,
//                CUBLAS_OP_N,
//                m - n / 2,
//                n - n / 2,
//                n / 2,
//                &tnegone,
//                A + n / 2,
//                cuda_data_type,
//                lda,

//                work,
//                cuda_data_type,
//                n / 2,

//                &tone,
//                W + n / 2,
//                cuda_data_type,
//                ldw,
//                cublas_compute_type,
//                CUBLAS_GEMM_DEFAULT);
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n - n / 2,
                n / 2,
                &tnegone,
                W,
                ldw,
                work,
                n / 2,
                &tone,
                W + n / 2 * ldw,
                ldw);
  g_Litter_GEMM_Time += stopTimer();

  return;
}

#if 0
int main(int argc, char *argv[])
{
    // 要求nb必须小于n，且n必须是nb的倍数
    long m, n, nb;

    const string fileName = "/home/wanghs/hpc/mySBR/Matrix_2048x2048.csv";
    vector<vector<double>> data = readMatrixFromFile(fileName);

    m = data.size();
    n = data[0].size();
    nb = 128;

    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;

    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);

    double *hA, *hW, *hR;

    hA = (double *)malloc(sizeof(double) * m * n);
    hW = (double *)malloc(sizeof(double) * m * n);
    hR = (double *)malloc(sizeof(double) * n * n);

    fillMatrix(hA, data);
    printMatrix(hA, m, n);

    double *A, *work, *R, *W;
    cudaMalloc(&A, sizeof(double) * m * n);
    cudaMalloc(&work, sizeof(double) * m * n);
    cudaMalloc(&R, sizeof(double) * n * n);
    cudaMalloc(&W, sizeof(double) * m * n);

    int *info;
    cudaMalloc(&info, sizeof(int));

    double *oriA;

    cudaMemcpy(A, hA, sizeof(double) * m * n, cudaMemcpyHostToDevice);

    CHECK(cudaGetLastError());

    double done = 1.0, dzero = 0.0, dnegone = -1.0;

    for (int i = 0; i < n; i += nb)
    {
        // 1、对矩阵进行一个panel的QR分解，调用panelQR函数
        // 此函数分解过后，A中存放的是Y矩阵，W中存放的是W矩阵，R中存放的是R矩阵
        panelQR(cusolver_handle, cublas_handle, m - i, nb, A + i + i * m, m,
                W + i + i * m, m, R + i + i * n, n, work, info);

        // 2、使用panel分解得到的W、Y更新尾矩阵A2=(I-WY')'*A2=(I-YW')A2=A2-YW'A2
        // 注意最后一次分解完成不用进行更新
        if (n - i > nb)
        {
            // 2.1 先计算W'A2,把结果存放到work中
            cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, nb, n - (i + nb),
                        m - i, &done, W + i + i * m, m, A + i + (i + nb) * m, m,
                        &dzero, work, nb);

            // 2.2 计算A2-YW'A2,也就是计算A2-Y*work
            cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m - i, n - (i + nb),
                        nb, &dnegone, A + i + i * m, m, work, nb, &done,
                        A + i + (i + nb) * m, m);

            // 2.3 将A的右上角部分拷贝到R中，同时对A的右上角部分清零
            dim3 gridDim((nb + 32 - 1) / 32, (n - (i + nb) + 32 - 1) / 32);
            dim3 blockDim(32, 32);

            copyAndClear<<<gridDim, blockDim>>>(nb, n - (i + nb), A + i + (i + nb) * m, m,
                                                R + i + (i + nb) * n, n);
        }

        // 3、求出新的W和Y，注意因为Y=[y1|y2],所以Y直接就已经求出来了，所以只用求W。
        // 因为W=[w1|w2-w1y1'w2],w1部分已经是求好的了，所以只需要求w2-w1y1'w2
        if (i > 0)
        {
            // 3.1 求y1'w2,把结果放到work中
            cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i, nb, m, &done, A,
                        m, W + 0 + i * m, m, &dzero, work, i);

            // 3.2 求w2-w1y1'w2, 也即是w2-w1*work
            cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, nb, i, &dnegone,
                        W, m, work, i, &done, W + 0 + i * m, m);
        }
    }
}
#endif