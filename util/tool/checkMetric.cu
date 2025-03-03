#include <cusolverDn.h>
#include <iostream>
#include "kernelOther.h"

// 先计算X=A'B
// 在计算||X||
void checkOrthogonality(cublasHandle_t cublas_handle, long M, long N, double *dA, long ldA,
                        double *dB, long ldB, double *work)
{
    long ldWork = M;
    double done = 1.0;
    double dzero = 0.0;

    cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, M, N,
                &done, dA, ldA, dB, ldB, &dzero, work, ldWork);

    dim3 gridDim((M + 31) / 32, (M + 31) / 32);
    dim3 blockDim(32, 32);
    launchKernel_IminusQ(gridDim, blockDim, M, M, work, ldWork);

    double sn;
    int incx = 1;

    cublasDnrm2(cublas_handle, M * M, work, incx, &sn);

    std::cout << "The Orthogonality metrix is " << sn / M << std::endl;
}