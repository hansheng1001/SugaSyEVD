#include <cusolverDn.h>


void computerQFromWY(cublasHandle_t cublas_handle, long M, long N, double *dQ, long ldQ,
                            double *dW, long ldW,double *dY, long ldY);