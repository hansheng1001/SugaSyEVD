#include <cusolverDn.h>


void computerWYFromSlide(cublasHandle_t cublas_handle, long M, long N, long slideWidth, double *dW, long ldW,
                         double *dY, long ldY, double *work);