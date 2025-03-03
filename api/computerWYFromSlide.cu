
#include "computerWYFromSlide.h"

// 从给出的条带化的W/Y中求出最后的W/Y
// Y就等于Y, Y_k+1=[Y_k|y_k+1]
// W=[W_k|w_k+1 - W_k*Y_k'*w_k+1]
void computerWYFromSlide(cublasHandle_t cublas_handle, long M, long N, long slideWidth, double *dW, long ldW,
                         double *dY, long ldY, double *work)
{
    long b = slideWidth;
    double done = 1.0;
    double dzero = 0.0;
    double dnegone = -1.0;

    long ldWork = M;

    // 求出W=[W_K|w_k+1 - W_K*Y_K'*w_k+1]
    for (long i = 2 * b; i <= N; i += b)
    {
        // 先计算出Y_K'*w_k+1, 把它放到work中
        // w_k+1的维度是Mxb,Y_K的维度为Mx(i-b)
        cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i - b, b, M,
                    &done, dY, ldY, dW + (i - b) * ldW, ldW, &dzero, work, ldWork);

        // 在计算出w_k+1 - W_K*Y_K'*w_k+1, 也就是w_k+1 - W_K*work
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, b, i - b,
                    &dnegone, dW, ldW, work, ldWork, &done, dW + (i - b) * ldW, ldW);
    }
}