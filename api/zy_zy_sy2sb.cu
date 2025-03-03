#include <iostream>
#include <string>
#include <vector>

#include "computerQFromWY.h"
#include "computerWYFromSlide.h"
#include "fileOpTool.h"
#include "kernelOther.h"
#include "myBase.h"
#include "PanelQR.h"
#include "TallShinnyQR.h"
#include "tc_ozimmu_syr2k.h"
#include <assert.h>

using namespace std;

float g_panelQR_time_ZY    = 0.0;
float g_tc_ozimmu_syr2k_ZY = 0.0;

float g_gemm_time_ZY = 0.0;
// 对对称矩阵进行操作SBR操作

// #define MY_DEBUG 1

template <typename T>
void my_ZY_ZY_SBR(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  T *dOriA,
                  long ldOriA,
                  T *dA,
                  long ldA,
                  T *dW,
                  long ldW,
                  T *dY,
                  long ldY,
                  T *dZ,
                  long ldZ,
                  T *dR,
                  long ldR,
                  T *work,
                  int *info);

template <>
void my_ZY_ZY_SBR(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  double *dOriA,
                  long ldOriA,
                  double *dA,
                  long ldA,
                  double *dW,
                  long ldW,
                  double *dY,
                  long ldY,
                  double *dZ,
                  long ldZ,
                  double *dR,
                  long ldR,
                  double *work,
                  int *info)
{
  // 此函数的结束条件
  if (0 >= M)
  {
    return;
  }

  // 验证条件
  if (0 != (M % nb))
  {
    cout << "M must be diviable by nb!" << endl;
    return;
  }

  // T *dwork;
  // cudaMalloc(&dwork, sizeof(T) * (M * nb));

  double done     = 1.0;
  double dzero    = 0.0;
  double dnegone  = -1.0;
  double dneghalf = -0.5;

  // 求出初始的OA的起始地址
  double *OA = dOriA + b * ldOriA + b;

  long ldWork = ldOriA;

  long i;
  for (i = b; (i <= nb) && (i < N); i += b)
  {
    // 对条带进行QR分解
    long m = M - i;
    long n = b;

    // dPanel是A[i+1][i-b+1]
    double *dPanel = dA + i + (i - b) * ldA;

    // dPanelW是W[i+1][i-b+1]
    double *dPanelW = dW + i + (i - b) * ldW;

    // R也是一个形状和A是一样的矩阵，dPanelR是R[i+1][i-b+1]
    double *dPanelR = dR + i + (i - b) * ldR;

#if MY_DEBUG
    cout << "print dPanelA:" << endl;
    printDeviceMatrixV2(dPanel, ldA, 32, 32);
#endif

    // cout << "print dPanelY 2:" << endl;
    // printDeviceMatrixV2(dA + i + (i - b) * ldA, ldA, m, n);

    // 对panel进行QR分解, dPanel中存放的是Y, dPanelW中存放的W，dPanelR中存放的是R
    startTimer();
    panelQR(cusolver_handle,
            cublas_handle,
            m,
            n,
            dPanel,
            ldA,
            dPanelW,
            ldW,
            dPanelR,
            ldR,
            work,
            info);
    // panelQR(cusolver_handle, cublas_handle, m, n, dA + i + (i - b) * ldA, ldA,
    //         dW + i + (i - b) * ldW, ldW, dR + i + (i - b) * ldR, ldR, work, info);
    g_panelQR_time_ZY += stopTimer();

    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

#if MY_DEBUG
    cout << "print dPanel:" << endl;
    printDeviceMatrixV2(dPanel, ldA, m, n);
    // printDeviceMatrixV2(dA + i + (i - b) * ldA, ldA, m, n);

    cout << "print dPanelW:" << endl;
    printDeviceMatrixV2(dPanelW, ldW, m, n);

    cout << "print dPanelR:" << endl;
    printDeviceMatrixV2(dPanelR, ldR, m, n);
#endif

    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    // 需要把A拷贝到Y中
    double *dPanelY = dY + i + (i - b) * ldY;
    launchKernel_copyMatrix(gridDim, blockDim, m, n, dPanel, ldA, dPanelY, ldY);

    // 不需要，panel中对于其左上角部分进行进行了清零了。
    // 只需要Y的下三角部分，得到真正的Y
    // getLower<<<gridDim, blockDim>>>(m, n, dPanelY, ldY);

    // 复制R到panelA的右上部分
    // 同时把panelA的左下部分置为0
    launchKernel_getU(gridDim, blockDim, m, n, dPanelR, ldR, dPanel, ldA);

#if MY_DEBUG
    cout << "print dPanelR:" << endl;
    printDeviceMatrixV2(dPanel, ldA, m, n);
#endif

    // 先注释掉,再最后使用
    // 将panelA复制到panelA的转置中
    // launchKernel_copyMatrixAToTranpB(gridDim, blockDim, m, n, dPanel, ldA, dA + (i - b) + i *
    // ldA, ldA);

#if MY_DEBUG
    cout << "print dPanel': " << endl;
    printDeviceMatrixV2(dA + (i - b) + i * ldA, ldA, n, m);
#endif

    // 求z
    // z:表示第几次ZY表示需要用到的z
    // y:表示第几次分解求出的y
    // w:表示第几次分解求出的w
    // Z:表示积攒出来的Z
    // Y:表示积攒出来的Y
    // A_i: 表示第i次分解后的尾矩阵, A_i = A - YZ' - ZY'
    // z = A_i*w- (1/2)yw'*A_i*w

    double *dPanelZ = dZ + i + (i - b) * ldZ;

    startTimer();
    if (i == b)
    {
      // 当i=0:表示第1次分解,这时候A_i = OA
      // z = OA*w- (1/2)yw'*OA*w
      // 1.1 计算OA*w
      // OA的维度为mxm;w的维度是mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    m,
      //                    &done,
      //                    OA,
      //                    CUDA_R_64F,
      //                    ldOriA,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              m,
      //              &done,
      //              OA,
      //              cuda_data_type,
      //              ldOriA,

      //              dPanelW,
      //              cuda_data_type,
      //              ldW,

      //              &dzero,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,

      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    m,
                    &done,
                    OA,

                    ldOriA,
 
                    dPanelW,

                    ldW,
 
                    &dzero,
                    dPanelZ,

                    ldZ);

      // 1.2 计算w'*OA*w, 假设OA*w是X,它是存放在dZ中的
      // w'的维度为bxm; X的维度是mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              b,
      //              b,
      //              m,
      //              &done,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);
      
      cublasDgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    b,
                    b,
                    m,
                    &done,
                    dPanelW,

                    ldW,
                    dPanelZ,

                    ldZ,
                    &dzero,
                    work,

                    ldWork);

      // 1.2 计算OA*w- (1/2)yw'*OA*w,
      // 假设w'*OA*w是B,它是存放在work中的; OA*w存放在dZ中
      // y的维度为mxb; B的维度是bxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    b,
      //                    &dneghalf,
      //                    dPanelY,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              b,
      //              &dneghalf,
      //              dPanelY,
      //              cuda_data_type,
      //              ldY,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    b,
                    &dneghalf,
                    dPanelY,

                    ldY,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,

                    ldZ);
    }
    else
    {
      // 当i != 0
      // z = A_i*w- (1/2)yw'*A_i*w
      // z = A_i*w- (1/2)yw'*A_i*w = (A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w

      // 2.1 计算(A - YZ' - ZY')*w = Aw - YZ'w - ZY'w

      // 2.1.1 计算Aw -> dPanelZ
      // w的维度为mxn = (M - i) x b = mxb
      // A的维度为(M-b)x(M-b), 有用的A的维度为(M-i) x (M-i)
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    m,
      //                    &done,
      //                    OA + (i - b) + (i - b) * ldOriA,
      //                    CUDA_R_64F,
      //                    ldOriA,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              m,
      //              &done,
      //              OA + (i - b) + (i - b) * ldOriA,
      //              cuda_data_type,
      //              ldOriA,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              &dzero,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    m,
                    &done,
                    OA + (i - b) + (i - b) * ldOriA,

                    ldOriA,
                    dPanelW,

                    ldW,
                    &dzero,
                    dPanelZ,

                    ldZ);

      // Aw - YZ'w
      // 2.1.2 计算Z'w -> work
      // Z'的维度为(i-b)x(M-b),缩减为(i-b)x(M-i)
      // w的维度为mxn = (M - i) x b =
      // mxb,可以扩展为(M-b)xb,但这样其上端全部是0,为了减少运算,所以不扩张
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    i - b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              i - b,
      //              b,
      //              m,
      //              &done,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);
      
      cublasDgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    i - b,
                    b,
                    m,
                    &done,
                    dZ + i,

                    ldZ,
                    dPanelW,

                    ldW,
                    &dzero,
                    work,

                    ldWork);
 

      // 2.1.3 计算Aw - YZ'w = dPanelZ - Y *work -> dPanelZ
      // Y的维度为(M-b)x(i-b),缩减为(M-i)x(i-b) = m x(i-b)
      // work的维度为(i-b)xb,
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    i - b,
      //                    &dnegone,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              i - b,
      //              &dnegone,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    i - b,
                    &dnegone,
                    dY + i,

                    ldY,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,
 
                    ldZ);

      // 3.计算Aw - YZ'w - ZY'w
      // 3.1 计算Y'w -> work
      // Y'的维度为(i-b)x(M-b),缩减为(i-b)x(M-i)
      // w的维度为mxn = (M - i) x b =
      // mxb,可以扩展为(M-b)xb,但这样其上端全部是0,为了减少运算,所以不扩张
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    i - b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              i - b,
      //              b,
      //              m,
      //              &done,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    i - b,
                    b,
                    m,
                    &done,
                    dY + i,

                    ldY,
                    dPanelW,

                    ldW,
                    &dzero,
                    work,

                    ldWork);

      // 3.2 计算Aw - YZ'w - ZY'w = dPanelZ - Z *work
      // Z的维度为(M-b)x(i-b),缩减为(M-i)x(i-b) = m x (i-b)
      // work的维度为(i-b)xb,
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    i - b,
      //                    &dnegone,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              i - b,
      //              &dnegone,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    i - b,
                    &dnegone,
                    dZ + i,

                    ldZ,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,

                    ldZ);

      // 4.计算(A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w
      // 4.1 计算w'*(A - YZ' - ZY')*w = w' * dPanelZ -> work
      // w'的维度为bxm
      // dPanelZ的维度为mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              b,
      //              b,
      //              m,
      //              &done,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    b,
                    b,
                    m,
                    &done,
                    dPanelW,

                    ldW,
                    dPanelZ,

                    ldZ,
                    &dzero,
                    work,

                    ldWork);

      // 4.2 计算(A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w -> dPanelZ
      // dPanelZ - (1/2)y*work
      // y的维度为mxb
      // work的维度为bxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    b,
      //                    &dneghalf,
      //                    dPanelY,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              b,
      //              &dneghalf,
      //              dPanelY,
      //              cuda_data_type,
      //              ldY,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    b,
                    &dneghalf,
                    dPanelY,

                    ldY,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,

                    ldZ);
    }

    // 6更新尾矩阵A的一部分GA
    // A = A - YZ' - ZY'
    // 注意只用更新A(i:M, i:i+b)
    if (i < nb)
    {
      // 6.1.计算A(i:M, i:i+b) - Y(i:M,)*Z(i:i+b,)' -> A
      // Y(i:M,)的维度为(M-i)xi=mxi
      // Z(i:i+b,)'的维度为ixb

      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_T,
      //                    m,
      //                    b,
      //                    i,
      //                    &dnegone,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &done,
      //                    dA + i + i * ldA,
      //                    CUDA_R_64F,
      //                    ldA,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_T,
      //              m,
      //              b,
      //              i,
      //              &dnegone,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              &done,
      //              dA + i + i * ldA,
      //              cuda_data_type,
      //              ldA,
      //              cuda_data_type,
      //              CUBLAS_GEMM_DEFAULT);

       cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    m,
                    b,
                    i,
                    &dnegone,
                    dY + i,

                    ldY,
                    dZ + i,

                    ldZ,
                    &done,
                    dA + i + i * ldA,

                    ldA);

      // 6.2.计算A(i:M, i:i+b) - Y(i:M,)*Z(i:i+b,)' - Z(i:M,)*Y(i:i+b,)' -> A - Z(i:M,)*Y(i:i+b,)'
      // Z(i:M,)的维度为(M-i)xi=mxi
      // Y(i:i+b,)'的维度为ixb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_T,
      //                    m,
      //                    b,
      //                    i,
      //                    &dnegone,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    &done,
      //                    dA + i + i * ldA,
      //                    CUDA_R_64F,
      //                    ldA,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_T,
      //              m,
      //              b,
      //              i,
      //              &dnegone,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              &done,
      //              dA + i + i * ldA,
      //              cuda_data_type,
      //              ldA,
      //              cuda_data_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    m,
                    b,
                    i,
                    &dnegone,
                    dZ + i,

                    ldZ,
                    dY + i,

                    ldY,
                    &done,
                    dA + i + i * ldA,

                    ldA);
    }

    g_gemm_time_ZY += stopTimer();

#if MY_DEBUG
    cout << "print dGA:" << endl;
    printDeviceMatrixV2(dA + b + i * ldA, ldA, M - b, b);
#endif
  }

  // dim3 gridDim3((M - b + 31) / 32, (i + 31) / 32);
  // dim3 blockDim3(32, 32);
  // launchKernel_copyMatrixAToTranpB(gridDim3, blockDim3, M - b, i, dA + b, ldA, dA + b * ldA,
  // ldA);

  if (0 >= N - nb)
  {
#if MY_DEBUG
    cout << "SBR end!" << endl;
#endif
    return;
  }

  // cudaFree(dwork);
  // cudaMalloc(&dwork, sizeof(double) * (M + nb) * (N + nb));

  // 更新其他的dA的部分
  // 使用ZY方式更新dA的其他部分
  // Z = A*W - 1/2(Y*W'*A*W)

  long lm = M - nb; // 注意在实际分解的情况下，肯定是M-b > nb的
  long ln = nb;

  dim3 block1(32, 32);
  dim3 grid1((lm + 31) / 32, (lm + 31) / 32);

  startTimer();
  //   if (lm > 32768)
  //   {
  //     tc_ozimmu_syr2k(cublas_handle,
  //                     lm,
  //                     ln,
  //                     dnegone,
  //                     dY + nb,
  //                     ldY,
  //                     dZ + nb,
  //                     ldZ,
  //                     done,
  //                     OA + (nb - b) + (nb - b) * ldOriA,
  //                     ldOriA,
  //                     nb);
  //     // OA, ldOriA, nb);
  //   }
  //   else
  //   {
  //     cublasDsyr2k(cublas_handle,
  //                  CUBLAS_FILL_MODE_LOWER,
  //                  CUBLAS_OP_N,
  //                  lm,
  //                  ln,
  //                  &dnegone,
  //                  dY + nb,
  //                  ldY,
  //                  dZ + nb,
  //                  ldZ,
  //                  &done,
  //                  OA + (nb - b) + (nb - b) * ldOriA,
  //                  ldOriA);
  //   }

  tc_ozimmu_syr2k(cublas_handle,
                  lm,
                  ln,
                  dnegone,
                  dY + nb,
                  ldY,
                  dZ + nb,
                  ldZ,
                  done,
                  OA + (nb - b) + (nb - b) * ldOriA,
                  ldOriA,
                  nb);
  g_tc_ozimmu_syr2k_ZY += stopTimer();

  launchKernel_CpyMatrixL2U(grid1, block1, lm, OA + (nb - b) + (nb - b) * ldOriA, ldOriA);

  // printf("OA:\n");
  // printDeviceMatrixV2(OA, ldOriA, 10, 10);

  // printf("OA end 10:\n");
  // printDeviceMatrixV2(OA + (lm - 10) + (lm - 10) * ldOriA, ldOriA, 10, 10);

  dim3 gridDim2((M - nb + 31) / 32, (N - nb + 31) / 32);
  dim3 blockDim2(32, 32);

  // cudaFree(dwork);

  // OA和dOriA为相同矩阵，相当于把OA拷贝到dA中
  launchKernel_copyMatrix(gridDim2,
                          blockDim2,
                          M - nb,
                          N - nb,
                          dOriA + nb + nb * ldOriA,
                          ldOriA,
                          dA + nb + nb * ldA,
                          ldA);

  M = N = M - nb;
  dA    = dA + nb + nb * ldA;
  // dW = dW + nb + nb * ldW;
  // dY = dY + nb + nb * ldY;
  // dR = dR + nb + nb * ldR;

  lm = M - b;
  dim3 grid2((lm + 31) / 32, (ln + 31) / 32);

  launchKernel_ClearMatrix(grid2, block1, lm, ln, dW + b, ldW);
  launchKernel_ClearMatrix(grid2, block1, lm, ln, dY + b, ldY);
  launchKernel_ClearMatrix(grid2, block1, lm, ln, dZ + b, ldZ);

  dW = dW + nb;
  dY = dY + nb;
  dR = dR + nb;

  dOriA = dOriA + nb + nb * ldOriA;

  // dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  // dim3 blockDim(32, 32);
  // // 把A中更新后的矩阵拷贝到A中
  // launchKernel_copyMatrix(gridDim, blockDim, M, N, dA, ldA, dOriA, ldOriA);

  // 迭代此方法
  my_ZY_ZY_SBR(cusolver_handle,
               cublas_handle,
               M,
               N,
               b,
               nb,
               dOriA,
               ldOriA,
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
               work,
               info);
}

template <>
void my_ZY_ZY_SBR(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  float *dOriA,
                  long ldOriA,
                  float *dA,
                  long ldA,
                  float *dW,
                  long ldW,
                  float *dY,
                  long ldY,
                  float *dZ,
                  long ldZ,
                  float *dR,
                  long ldR,
                  float *work,
                  int *info)
{
  // 此函数的结束条件
  if (0 >= M)
  {
    return;
  }

  // 验证条件
  if (0 != (M % nb))
  {
    cout << "M must be diviable by nb!" << endl;
    return;
  }

  // T *dwork;
  // cudaMalloc(&dwork, sizeof(T) * (M * nb));

  float done     = 1.0;
  float dzero    = 0.0;
  float dnegone  = -1.0;
  float dneghalf = -0.5;

  // 求出初始的OA的起始地址
  float *OA = dOriA + b * ldOriA + b;

  long ldWork = ldOriA;

  long i;
  for (i = b; (i <= nb) && (i < N); i += b)
  {
    // 对条带进行QR分解
    long m = M - i;
    long n = b;

    // dPanel是A[i+1][i-b+1]
    float *dPanel = dA + i + (i - b) * ldA;

    // dPanelW是W[i+1][i-b+1]
    float *dPanelW = dW + i + (i - b) * ldW;

    // R也是一个形状和A是一样的矩阵，dPanelR是R[i+1][i-b+1]
    float *dPanelR = dR + i + (i - b) * ldR;

#if MY_DEBUG
    cout << "print dPanelA:" << endl;
    printDeviceMatrixV2(dPanel, ldA, 32, 32);
#endif

    // cout << "print dPanelY 2:" << endl;
    // printDeviceMatrixV2(dA + i + (i - b) * ldA, ldA, m, n);

    // 对panel进行QR分解, dPanel中存放的是Y, dPanelW中存放的W，dPanelR中存放的是R
    startTimer();
    panelQR(cusolver_handle,
            cublas_handle,
            m,
            n,
            dPanel,
            ldA,
            dPanelW,
            ldW,
            dPanelR,
            ldR,
            work,
            info);
    // panelQR(cusolver_handle, cublas_handle, m, n, dA + i + (i - b) * ldA, ldA,
    //         dW + i + (i - b) * ldW, ldW, dR + i + (i - b) * ldR, ldR, work, info);
    g_panelQR_time_ZY += stopTimer();

    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

#if MY_DEBUG
    cout << "print dPanel:" << endl;
    printDeviceMatrixV2(dPanel, ldA, m, n);
    // printDeviceMatrixV2(dA + i + (i - b) * ldA, ldA, m, n);

    cout << "print dPanelW:" << endl;
    printDeviceMatrixV2(dPanelW, ldW, m, n);

    cout << "print dPanelR:" << endl;
    printDeviceMatrixV2(dPanelR, ldR, m, n);
#endif

    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    // 需要把A拷贝到Y中
    float *dPanelY = dY + i + (i - b) * ldY;
    launchKernel_copyMatrix(gridDim, blockDim, m, n, dPanel, ldA, dPanelY, ldY);

    // 不需要，panel中对于其左上角部分进行进行了清零了。
    // 只需要Y的下三角部分，得到真正的Y
    // getLower<<<gridDim, blockDim>>>(m, n, dPanelY, ldY);

    // 复制R到panelA的右上部分
    // 同时把panelA的左下部分置为0
    launchKernel_getU(gridDim, blockDim, m, n, dPanelR, ldR, dPanel, ldA);

#if MY_DEBUG
    cout << "print dPanelR:" << endl;
    printDeviceMatrixV2(dPanel, ldA, m, n);
#endif

    // 先注释掉,再最后使用
    // 将panelA复制到panelA的转置中
    // launchKernel_copyMatrixAToTranpB(gridDim, blockDim, m, n, dPanel, ldA, dA + (i - b) + i *
    // ldA, ldA);

#if MY_DEBUG
    cout << "print dPanel': " << endl;
    printDeviceMatrixV2(dA + (i - b) + i * ldA, ldA, n, m);
#endif

    // 求z
    // z:表示第几次ZY表示需要用到的z
    // y:表示第几次分解求出的y
    // w:表示第几次分解求出的w
    // Z:表示积攒出来的Z
    // Y:表示积攒出来的Y
    // A_i: 表示第i次分解后的尾矩阵, A_i = A - YZ' - ZY'
    // z = A_i*w- (1/2)yw'*A_i*w

    float *dPanelZ = dZ + i + (i - b) * ldZ;

    startTimer();
    if (i == b)
    {
      // 当i=0:表示第1次分解,这时候A_i = OA
      // z = OA*w- (1/2)yw'*OA*w
      // 1.1 计算OA*w
      // OA的维度为mxm;w的维度是mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    m,
      //                    &done,
      //                    OA,
      //                    CUDA_R_64F,
      //                    ldOriA,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              m,
      //              &done,
      //              OA,
      //              cuda_data_type,
      //              ldOriA,

      //              dPanelW,
      //              cuda_data_type,
      //              ldW,

      //              &dzero,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,

      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    m,
                    &done,
                    OA,

                    ldOriA,
 
                    dPanelW,

                    ldW,
 
                    &dzero,
                    dPanelZ,

                    ldZ);

      // 1.2 计算w'*OA*w, 假设OA*w是X,它是存放在dZ中的
      // w'的维度为bxm; X的维度是mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              b,
      //              b,
      //              m,
      //              &done,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);
      
      cublasSgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    b,
                    b,
                    m,
                    &done,
                    dPanelW,

                    ldW,
                    dPanelZ,

                    ldZ,
                    &dzero,
                    work,

                    ldWork);

      // 1.2 计算OA*w- (1/2)yw'*OA*w,
      // 假设w'*OA*w是B,它是存放在work中的; OA*w存放在dZ中
      // y的维度为mxb; B的维度是bxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    b,
      //                    &dneghalf,
      //                    dPanelY,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              b,
      //              &dneghalf,
      //              dPanelY,
      //              cuda_data_type,
      //              ldY,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    b,
                    &dneghalf,
                    dPanelY,

                    ldY,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,

                    ldZ);
    }
    else
    {
      // 当i != 0
      // z = A_i*w- (1/2)yw'*A_i*w
      // z = A_i*w- (1/2)yw'*A_i*w = (A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w

      // 2.1 计算(A - YZ' - ZY')*w = Aw - YZ'w - ZY'w

      // 2.1.1 计算Aw -> dPanelZ
      // w的维度为mxn = (M - i) x b = mxb
      // A的维度为(M-b)x(M-b), 有用的A的维度为(M-i) x (M-i)
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    m,
      //                    &done,
      //                    OA + (i - b) + (i - b) * ldOriA,
      //                    CUDA_R_64F,
      //                    ldOriA,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              m,
      //              &done,
      //              OA + (i - b) + (i - b) * ldOriA,
      //              cuda_data_type,
      //              ldOriA,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              &dzero,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    m,
                    &done,
                    OA + (i - b) + (i - b) * ldOriA,

                    ldOriA,
                    dPanelW,

                    ldW,
                    &dzero,
                    dPanelZ,

                    ldZ);

      // Aw - YZ'w
      // 2.1.2 计算Z'w -> work
      // Z'的维度为(i-b)x(M-b),缩减为(i-b)x(M-i)
      // w的维度为mxn = (M - i) x b =
      // mxb,可以扩展为(M-b)xb,但这样其上端全部是0,为了减少运算,所以不扩张
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    i - b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              i - b,
      //              b,
      //              m,
      //              &done,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);
      
      cublasSgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    i - b,
                    b,
                    m,
                    &done,
                    dZ + i,

                    ldZ,
                    dPanelW,

                    ldW,
                    &dzero,
                    work,

                    ldWork);
 

      // 2.1.3 计算Aw - YZ'w = dPanelZ - Y *work -> dPanelZ
      // Y的维度为(M-b)x(i-b),缩减为(M-i)x(i-b) = m x(i-b)
      // work的维度为(i-b)xb,
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    i - b,
      //                    &dnegone,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              i - b,
      //              &dnegone,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    i - b,
                    &dnegone,
                    dY + i,

                    ldY,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,
 
                    ldZ);

      // 3.计算Aw - YZ'w - ZY'w
      // 3.1 计算Y'w -> work
      // Y'的维度为(i-b)x(M-b),缩减为(i-b)x(M-i)
      // w的维度为mxn = (M - i) x b =
      // mxb,可以扩展为(M-b)xb,但这样其上端全部是0,为了减少运算,所以不扩张
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    i - b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              i - b,
      //              b,
      //              m,
      //              &done,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    i - b,
                    b,
                    m,
                    &done,
                    dY + i,

                    ldY,
                    dPanelW,

                    ldW,
                    &dzero,
                    work,

                    ldWork);

      // 3.2 计算Aw - YZ'w - ZY'w = dPanelZ - Z *work
      // Z的维度为(M-b)x(i-b),缩减为(M-i)x(i-b) = m x (i-b)
      // work的维度为(i-b)xb,
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    i - b,
      //                    &dnegone,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              i - b,
      //              &dnegone,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    i - b,
                    &dnegone,
                    dZ + i,

                    ldZ,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,

                    ldZ);

      // 4.计算(A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w
      // 4.1 计算w'*(A - YZ' - ZY')*w = w' * dPanelZ -> work
      // w'的维度为bxm
      // dPanelZ的维度为mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_T,
      //              CUBLAS_OP_N,
      //              b,
      //              b,
      //              m,
      //              &done,
      //              dPanelW,
      //              cuda_data_type,
      //              ldW,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              &dzero,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    b,
                    b,
                    m,
                    &done,
                    dPanelW,

                    ldW,
                    dPanelZ,

                    ldZ,
                    &dzero,
                    work,

                    ldWork);

      // 4.2 计算(A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w -> dPanelZ
      // dPanelZ - (1/2)y*work
      // y的维度为mxb
      // work的维度为bxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    b,
      //                    &dneghalf,
      //                    dPanelY,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_N,
      //              m,
      //              b,
      //              b,
      //              &dneghalf,
      //              dPanelY,
      //              cuda_data_type,
      //              ldY,
      //              work,
      //              cuda_data_type,
      //              ldWork,
      //              &done,
      //              dPanelZ,
      //              cuda_data_type,
      //              ldZ,
      //              cublas_compute_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    b,
                    b,
                    &dneghalf,
                    dPanelY,

                    ldY,
                    work,

                    ldWork,
                    &done,
                    dPanelZ,

                    ldZ);
    }

    // 6更新尾矩阵A的一部分GA
    // A = A - YZ' - ZY'
    // 注意只用更新A(i:M, i:i+b)
    if (i < nb)
    {
      // 6.1.计算A(i:M, i:i+b) - Y(i:M,)*Z(i:i+b,)' -> A
      // Y(i:M,)的维度为(M-i)xi=mxi
      // Z(i:i+b,)'的维度为ixb

      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_T,
      //                    m,
      //                    b,
      //                    i,
      //                    &dnegone,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &done,
      //                    dA + i + i * ldA,
      //                    CUDA_R_64F,
      //                    ldA,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_T,
      //              m,
      //              b,
      //              i,
      //              &dnegone,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              &done,
      //              dA + i + i * ldA,
      //              cuda_data_type,
      //              ldA,
      //              cuda_data_type,
      //              CUBLAS_GEMM_DEFAULT);

       cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    m,
                    b,
                    i,
                    &dnegone,
                    dY + i,

                    ldY,
                    dZ + i,

                    ldZ,
                    &done,
                    dA + i + i * ldA,

                    ldA);

      // 6.2.计算A(i:M, i:i+b) - Y(i:M,)*Z(i:i+b,)' - Z(i:M,)*Y(i:i+b,)' -> A - Z(i:M,)*Y(i:i+b,)'
      // Z(i:M,)的维度为(M-i)xi=mxi
      // Y(i:i+b,)'的维度为ixb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_T,
      //                    m,
      //                    b,
      //                    i,
      //                    &dnegone,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    &done,
      //                    dA + i + i * ldA,
      //                    CUDA_R_64F,
      //                    ldA,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      // cublasGemmEx(cublas_handle,
      //              CUBLAS_OP_N,
      //              CUBLAS_OP_T,
      //              m,
      //              b,
      //              i,
      //              &dnegone,
      //              dZ + i,
      //              cuda_data_type,
      //              ldZ,
      //              dY + i,
      //              cuda_data_type,
      //              ldY,
      //              &done,
      //              dA + i + i * ldA,
      //              cuda_data_type,
      //              ldA,
      //              cuda_data_type,
      //              CUBLAS_GEMM_DEFAULT);

      cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    m,
                    b,
                    i,
                    &dnegone,
                    dZ + i,

                    ldZ,
                    dY + i,

                    ldY,
                    &done,
                    dA + i + i * ldA,

                    ldA);
    }

    g_gemm_time_ZY += stopTimer();

#if MY_DEBUG
    cout << "print dGA:" << endl;
    printDeviceMatrixV2(dA + b + i * ldA, ldA, M - b, b);
#endif
  }

  // dim3 gridDim3((M - b + 31) / 32, (i + 31) / 32);
  // dim3 blockDim3(32, 32);
  // launchKernel_copyMatrixAToTranpB(gridDim3, blockDim3, M - b, i, dA + b, ldA, dA + b * ldA,
  // ldA);

  if (0 >= N - nb)
  {
#if MY_DEBUG
    cout << "SBR end!" << endl;
#endif
    return;
  }

  // cudaFree(dwork);
  // cudaMalloc(&dwork, sizeof(double) * (M + nb) * (N + nb));

  // 更新其他的dA的部分
  // 使用ZY方式更新dA的其他部分
  // Z = A*W - 1/2(Y*W'*A*W)

  long lm = M - nb; // 注意在实际分解的情况下，肯定是M-b > nb的
  long ln = nb;

  dim3 block1(32, 32);
  dim3 grid1((lm + 31) / 32, (lm + 31) / 32);

  startTimer();
  //   if (lm > 32768)
  //   {
  //     tc_ozimmu_syr2k(cublas_handle,
  //                     lm,
  //                     ln,
  //                     dnegone,
  //                     dY + nb,
  //                     ldY,
  //                     dZ + nb,
  //                     ldZ,
  //                     done,
  //                     OA + (nb - b) + (nb - b) * ldOriA,
  //                     ldOriA,
  //                     nb);
  //     // OA, ldOriA, nb);
  //   }
  //   else
  //   {
  //     cublasDsyr2k(cublas_handle,
  //                  CUBLAS_FILL_MODE_LOWER,
  //                  CUBLAS_OP_N,
  //                  lm,
  //                  ln,
  //                  &dnegone,
  //                  dY + nb,
  //                  ldY,
  //                  dZ + nb,
  //                  ldZ,
  //                  &done,
  //                  OA + (nb - b) + (nb - b) * ldOriA,
  //                  ldOriA);
  //   }

  tc_ozimmu_syr2k(cublas_handle,
                  lm,
                  ln,
                  dnegone,
                  dY + nb,
                  ldY,
                  dZ + nb,
                  ldZ,
                  done,
                  OA + (nb - b) + (nb - b) * ldOriA,
                  ldOriA,
                  nb);
  g_tc_ozimmu_syr2k_ZY += stopTimer();

  launchKernel_CpyMatrixL2U(grid1, block1, lm, OA + (nb - b) + (nb - b) * ldOriA, ldOriA);

  // printf("OA:\n");
  // printDeviceMatrixV2(OA, ldOriA, 10, 10);

  // printf("OA end 10:\n");
  // printDeviceMatrixV2(OA + (lm - 10) + (lm - 10) * ldOriA, ldOriA, 10, 10);

  dim3 gridDim2((M - nb + 31) / 32, (N - nb + 31) / 32);
  dim3 blockDim2(32, 32);

  // cudaFree(dwork);

  // OA和dOriA为相同矩阵，相当于把OA拷贝到dA中
  launchKernel_copyMatrix(gridDim2,
                          blockDim2,
                          M - nb,
                          N - nb,
                          dOriA + nb + nb * ldOriA,
                          ldOriA,
                          dA + nb + nb * ldA,
                          ldA);

  M = N = M - nb;
  dA    = dA + nb + nb * ldA;
  // dW = dW + nb + nb * ldW;
  // dY = dY + nb + nb * ldY;
  // dR = dR + nb + nb * ldR;

  lm = M - b;
  dim3 grid2((lm + 31) / 32, (ln + 31) / 32);

  launchKernel_ClearMatrix(grid2, block1, lm, ln, dW + b, ldW);
  launchKernel_ClearMatrix(grid2, block1, lm, ln, dY + b, ldY);
  launchKernel_ClearMatrix(grid2, block1, lm, ln, dZ + b, ldZ);

  dW = dW + nb;
  dY = dY + nb;
  dR = dR + nb;

  dOriA = dOriA + nb + nb * ldOriA;

  // dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  // dim3 blockDim(32, 32);
  // // 把A中更新后的矩阵拷贝到A中
  // launchKernel_copyMatrix(gridDim, blockDim, M, N, dA, ldA, dOriA, ldOriA);

  // 迭代此方法
  my_ZY_ZY_SBR(cusolver_handle,
               cublas_handle,
               M,
               N,
               b,
               nb,
               dOriA,
               ldOriA,
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
               work,
               info);
}


template <typename T>
void my_ZY_ZY_SBR_Vector(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  T *dOriA,
                  long ldOriA,
                  T *dA,
                  long ldA,
                  T *dW,
                  long ldW,
                  T *dY,
                  long ldY,
                  T *dZ,
                  long ldZ,
                  T *dR,
                  long ldR,
                  T *work,
                  int *info)
{
  // 此函数的结束条件
  if (0 >= M)
  {
    return;
  }

  // 验证条件
  if (0 != (M % nb))
  {
    cout << "M must be diviable by nb!" << endl;
    return;
  }

  // T *dwork;
  // cudaMalloc(&dwork, sizeof(T) * (M * nb));

  T done     = 1.0;
  T dzero    = 0.0;
  T dnegone  = -1.0;
  T dneghalf = -0.5;

  // 求出初始的OA的起始地址
  T *OA = dOriA + b * ldOriA + b;

  long ldWork = ldOriA;

  long i;
  for (i = b; (i <= nb) && (i < N); i += b)
  {
    // 对条带进行QR分解
    long m = M - i;
    long n = b;

    // dPanel是A[i+1][i-b+1]
    T *dPanel = dA + i + (i - b) * ldA;

    // dPanelW是W[i+1][i-b+1]
    T *dPanelW = dW + i + (i - b) * ldW;

    // R也是一个形状和A是一样的矩阵，dPanelR是R[i+1][i-b+1]
    T *dPanelR = dR + i + (i - b) * ldR;

#if MY_DEBUG
    cout << "print dPanelA:" << endl;
    printDeviceMatrixV2(dPanel, ldA, 32, 32);
#endif

    // cout << "print dPanelY 2:" << endl;
    // printDeviceMatrixV2(dA + i + (i - b) * ldA, ldA, m, n);

    // 对panel进行QR分解, dPanel中存放的是Y, dPanelW中存放的W，dPanelR中存放的是R
    startTimer();
    panelQR(cusolver_handle,
            cublas_handle,
            m,
            n,
            dPanel,
            ldA,
            dPanelW,
            ldW,
            dPanelR,
            ldR,
            work,
            info);
    // panelQR(cusolver_handle, cublas_handle, m, n, dA + i + (i - b) * ldA, ldA,
    //         dW + i + (i - b) * ldW, ldW, dR + i + (i - b) * ldR, ldR, work, info);
    g_panelQR_time_ZY += stopTimer();

    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

#if MY_DEBUG
    cout << "print dPanel:" << endl;
    printDeviceMatrixV2(dPanel, ldA, m, n);
    // printDeviceMatrixV2(dA + i + (i - b) * ldA, ldA, m, n);

    cout << "print dPanelW:" << endl;
    printDeviceMatrixV2(dPanelW, ldW, m, n);

    cout << "print dPanelR:" << endl;
    printDeviceMatrixV2(dPanelR, ldR, m, n);
#endif

    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    // 需要把A拷贝到Y中
    T *dPanelY = dY + i + (i - b) * ldY;
    launchKernel_copyMatrix(gridDim, blockDim, m, n, dPanel, ldA, dPanelY, ldY);

    // 不需要，panel中对于其左上角部分进行进行了清零了。
    // 只需要Y的下三角部分，得到真正的Y
    // getLower<<<gridDim, blockDim>>>(m, n, dPanelY, ldY);

    // 复制R到panelA的右上部分
    // 同时把panelA的左下部分置为0
    launchKernel_getU(gridDim, blockDim, m, n, dPanelR, ldR, dPanel, ldA);

#if MY_DEBUG
    cout << "print dPanelR:" << endl;
    printDeviceMatrixV2(dPanel, ldA, m, n);
#endif

    // 先注释掉,再最后使用
    // 将panelA复制到panelA的转置中
    // launchKernel_copyMatrixAToTranpB(gridDim, blockDim, m, n, dPanel, ldA, dA + (i - b) + i *
    // ldA, ldA);

#if MY_DEBUG
    cout << "print dPanel': " << endl;
    printDeviceMatrixV2(dA + (i - b) + i * ldA, ldA, n, m);
#endif

    // 求z
    // z:表示第几次ZY表示需要用到的z
    // y:表示第几次分解求出的y
    // w:表示第几次分解求出的w
    // Z:表示积攒出来的Z
    // Y:表示积攒出来的Y
    // A_i: 表示第i次分解后的尾矩阵, A_i = A - YZ' - ZY'
    // z = A_i*w- (1/2)yw'*A_i*w

    T *dPanelZ = dZ + i + (i - b) * ldZ;

    startTimer();
    if (i == b)
    {
      // 当i=0:表示第1次分解,这时候A_i = OA
      // z = OA*w- (1/2)yw'*OA*w
      // 1.1 计算OA*w
      // OA的维度为mxm;w的维度是mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    m,
      //                    &done,
      //                    OA,
      //                    CUDA_R_64F,
      //                    ldOriA,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   m,
                   &done,
                   OA,

                   ldOriA,

                   dPanelW,

                   ldW,

                   &dzero,
                   dPanelZ,

                   ldZ);

      // 1.2 计算w'*OA*w, 假设OA*w是X,它是存放在dZ中的
      // w'的维度为bxm; X的维度是mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   b,
                   b,
                   m,
                   &done,
                   dPanelW,

                   ldW,
                   dPanelZ,

                   ldZ,
                   &dzero,
                   work,

                   ldWork);

      // 1.2 计算OA*w- (1/2)yw'*OA*w,
      // 假设w'*OA*w是B,它是存放在work中的; OA*w存放在dZ中
      // y的维度为mxb; B的维度是bxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    b,
      //                    &dneghalf,
      //                    dPanelY,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   b,
                   &dneghalf,
                   dPanelY,

                   ldY,
                   work,

                   ldWork,
                   &done,
                   dPanelZ,

                   ldZ);
    }
    else
    {
      // 当i != 0
      // z = A_i*w- (1/2)yw'*A_i*w
      // z = A_i*w- (1/2)yw'*A_i*w = (A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w

      // 2.1 计算(A - YZ' - ZY')*w = Aw - YZ'w - ZY'w

      // 2.1.1 计算Aw -> dPanelZ
      // w的维度为mxn = (M - i) x b = mxb
      // A的维度为(M-b)x(M-b), 有用的A的维度为(M-i) x (M-i)
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    m,
      //                    &done,
      //                    OA + (i - b) + (i - b) * ldOriA,
      //                    CUDA_R_64F,
      //                    ldOriA,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   m,
                   &done,
                   OA + (i - b) + (i - b) * ldOriA,

                   ldOriA,
                   dPanelW,

                   ldW,
                   &dzero,
                   dPanelZ,

                   ldZ);

      // Aw - YZ'w
      // 2.1.2 计算Z'w -> work
      // Z'的维度为(i-b)x(M-b),缩减为(i-b)x(M-i)
      // w的维度为mxn = (M - i) x b =
      // mxb,可以扩展为(M-b)xb,但这样其上端全部是0,为了减少运算,所以不扩张
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    i - b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   i - b,
                   b,
                   m,
                   &done,
                   dZ + i,

                   ldZ,
                   dPanelW,

                   ldW,
                   &dzero,
                   work,

                   ldWork);

      // 2.1.3 计算Aw - YZ'w = dPanelZ - Y *work -> dPanelZ
      // Y的维度为(M-b)x(i-b),缩减为(M-i)x(i-b) = m x(i-b)
      // work的维度为(i-b)xb,
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    i - b,
      //                    &dnegone,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   i - b,
                   &dnegone,
                   dY + i,

                   ldY,
                   work,

                   ldWork,
                   &done,
                   dPanelZ,

                   ldZ);

      // 3.计算Aw - YZ'w - ZY'w
      // 3.1 计算Y'w -> work
      // Y'的维度为(i-b)x(M-b),缩减为(i-b)x(M-i)
      // w的维度为mxn = (M - i) x b =
      // mxb,可以扩展为(M-b)xb,但这样其上端全部是0,为了减少运算,所以不扩张
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    i - b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   i - b,
                   b,
                   m,
                   &done,
                   dY + i,

                   ldY,
                   dPanelW,

                   ldW,
                   &dzero,
                   work,

                   ldWork);

      // 3.2 计算Aw - YZ'w - ZY'w = dPanelZ - Z *work
      // Z的维度为(M-b)x(i-b),缩减为(M-i)x(i-b) = m x (i-b)
      // work的维度为(i-b)xb,
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    i - b,
      //                    &dnegone,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   i - b,
                   &dnegone,
                   dZ + i,

                   ldZ,
                   work,

                   ldWork,
                   &done,
                   dPanelZ,

                   ldZ);

      // 4.计算(A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w
      // 4.1 计算w'*(A - YZ' - ZY')*w = w' * dPanelZ -> work
      // w'的维度为bxm
      // dPanelZ的维度为mxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    b,
      //                    b,
      //                    m,
      //                    &done,
      //                    dPanelW,
      //                    CUDA_R_64F,
      //                    ldW,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &dzero,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   b,
                   b,
                   m,
                   &done,
                   dPanelW,

                   ldW,
                   dPanelZ,

                   ldZ,
                   &dzero,
                   work,

                   ldWork);

      // 4.2 计算(A - YZ' - ZY')*w - (1/2)yw'*(A - YZ' - ZY')*w -> dPanelZ
      // dPanelZ - (1/2)y*work
      // y的维度为mxb
      // work的维度为bxb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_N,
      //                    m,
      //                    b,
      //                    b,
      //                    &dneghalf,
      //                    dPanelY,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    work,
      //                    CUDA_R_64F,
      //                    ldWork,
      //                    &done,
      //                    dPanelZ,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   b,
                   &dneghalf,
                   dPanelY,

                   ldY,
                   work,

                   ldWork,
                   &done,
                   dPanelZ,

                   ldZ);
    }

    // 6更新尾矩阵A的一部分GA
    // A = A - YZ' - ZY'
    // 注意只用更新A(i:M, i:i+b)
    if (i < nb)
    {
      // 6.1.计算A(i:M, i:i+b) - Y(i:M,)*Z(i:i+b,)' -> A
      // Y(i:M,)的维度为(M-i)xi=mxi
      // Z(i:i+b,)'的维度为ixb

      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_T,
      //                    m,
      //                    b,
      //                    i,
      //                    &dnegone,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    &done,
      //                    dA + i + i * ldA,
      //                    CUDA_R_64F,
      //                    ldA,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   b,
                   i,
                   &dnegone,
                   dY + i,

                   ldY,
                   dZ + i,

                   ldZ,
                   &done,
                   dA + i + i * ldA,

                   ldA);

      // 6.2.计算A(i:M, i:i+b) - Y(i:M,)*Z(i:i+b,)' - Z(i:M,)*Y(i:i+b,)' -> A - Z(i:M,)*Y(i:i+b,)'
      // Z(i:M,)的维度为(M-i)xi=mxi
      // Y(i:i+b,)'的维度为ixb
      //       cublasGemmEx(cublas_handle,
      //                    CUBLAS_OP_N,
      //                    CUBLAS_OP_T,
      //                    m,
      //                    b,
      //                    i,
      //                    &dnegone,
      //                    dZ + i,
      //                    CUDA_R_64F,
      //                    ldZ,
      //                    dY + i,
      //                    CUDA_R_64F,
      //                    ldY,
      //                    &done,
      //                    dA + i + i * ldA,
      //                    CUDA_R_64F,
      //                    ldA,
      //                    CUDA_R_64F,
      //                    CUBLAS_GEMM_DEFAULT);

      cublasDgemm(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   b,
                   i,
                   &dnegone,
                   dZ + i,

                   ldZ,
                   dY + i,

                   ldY,
                   &done,
                   dA + i + i * ldA,

                   ldA);
    }

    g_gemm_time_ZY += stopTimer();

#if MY_DEBUG
    cout << "print dGA:" << endl;
    printDeviceMatrixV2(dA + b + i * ldA, ldA, M - b, b);
#endif
  }

  // dim3 gridDim3((M - b + 31) / 32, (i + 31) / 32);
  // dim3 blockDim3(32, 32);
  // launchKernel_copyMatrixAToTranpB(gridDim3, blockDim3, M - b, i, dA + b, ldA, dA + b * ldA,
  // ldA);

  if (0 >= N - nb)
  {
#if MY_DEBUG
    cout << "SBR end!" << endl;
#endif
    return;
  }

  // cudaFree(dwork);
  // cudaMalloc(&dwork, sizeof(double) * (M + nb) * (N + nb));

  // 更新其他的dA的部分
  // 使用ZY方式更新dA的其他部分
  // Z = A*W - 1/2(Y*W'*A*W)

  long lm = M - nb; // 注意在实际分解的情况下，肯定是M-b > nb的
  long ln = nb;

  dim3 block1(32, 32);
  dim3 grid1((lm + 31) / 32, (lm + 31) / 32);

  startTimer();
  //   if (lm > 32768)
  //   {
  //     tc_ozimmu_syr2k(cublas_handle,
  //                     lm,
  //                     ln,
  //                     dnegone,
  //                     dY + nb,
  //                     ldY,
  //                     dZ + nb,
  //                     ldZ,
  //                     done,
  //                     OA + (nb - b) + (nb - b) * ldOriA,
  //                     ldOriA,
  //                     nb);
  //     // OA, ldOriA, nb);
  //   }
  //   else
  //   {
  //     cublasDsyr2k(cublas_handle,
  //                  CUBLAS_FILL_MODE_LOWER,
  //                  CUBLAS_OP_N,
  //                  lm,
  //                  ln,
  //                  &dnegone,
  //                  dY + nb,
  //                  ldY,
  //                  dZ + nb,
  //                  ldZ,
  //                  &done,
  //                  OA + (nb - b) + (nb - b) * ldOriA,
  //                  ldOriA);
  //   }

  tc_ozimmu_syr2k(cublas_handle,
                  lm,
                  ln,
                  dnegone,
                  dY + nb,
                  ldY,
                  dZ + nb,
                  ldZ,
                  done,
                  OA + (nb - b) + (nb - b) * ldOriA,
                  ldOriA,
                  nb);
  g_tc_ozimmu_syr2k_ZY += stopTimer();

  launchKernel_CpyMatrixL2U(grid1, block1, lm, OA + (nb - b) + (nb - b) * ldOriA, ldOriA);

  // printf("OA:\n");
  // printDeviceMatrixV2(OA, ldOriA, 10, 10);

  // printf("OA end 10:\n");
  // printDeviceMatrixV2(OA + (lm - 10) + (lm - 10) * ldOriA, ldOriA, 10, 10);

  dim3 gridDim2((M - nb + 31) / 32, (N - nb + 31) / 32);
  dim3 blockDim2(32, 32);

  // cudaFree(dwork);

  // OA和dOriA为相同矩阵，相当于把OA拷贝到dA中
  launchKernel_copyMatrix(gridDim2,
                          blockDim2,
                          M - nb,
                          N - nb,
                          dOriA + nb + nb * ldOriA,
                          ldOriA,
                          dA + nb + nb * ldA,
                          ldA);

  M = N = M - nb;
  dA    = dA + nb + nb * ldA;
  dW = dW + nb + nb * ldW;
  dY = dY + nb + nb * ldY;
  // dR = dR + nb + nb * ldR;

  lm = M - b;
  dim3 grid2((lm + 31) / 32, (ln + 31) / 32);

  // launchKernel_ClearMatrix(grid2, block1, lm, ln, dW + b, ldW);
  // launchKernel_ClearMatrix(grid2, block1, lm, ln, dY + b, ldY);
  launchKernel_ClearMatrix(grid2, block1, lm, ln, dZ + b, ldZ);

  // dW = dW + nb;
  // dY = dY + nb;
  dR = dR + nb;

  dOriA = dOriA + nb + nb * ldOriA;

  // dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  // dim3 blockDim(32, 32);
  // // 把A中更新后的矩阵拷贝到A中
  // launchKernel_copyMatrix(gridDim, blockDim, M, N, dA, ldA, dOriA, ldOriA);

  // 迭代此方法
  my_ZY_ZY_SBR_Vector(cusolver_handle,
               cublas_handle,
               M,
               N,
               b,
               nb,
               dOriA,
               ldOriA,
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
               work,
               info);
}

template void my_ZY_ZY_SBR_Vector(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  double *dOriA,
                  long ldOriA,
                  double *dA,
                  long ldA,
                  double *dW,
                  long ldW,
                  double *dY,
                  long ldY,
                  double *dZ,
                  long ldZ,
                  double *dR,
                  long ldR,
                  double *work,
                  int *info);
