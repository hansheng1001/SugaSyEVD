#include <cusolverDn.h>

void checkOrthogonality(cublasHandle_t cublas_handle, long M, long N, double *dA, long ldA,
                        double *dB, long ldB, double *work);