
#pragma once

#include <cusolverDn.h>

// void panelQR(cusolverDnHandle_t cusolver_handle,
//              cublasHandle_t cublas_handle,
//              long m,
//              long n,
//              double *A,
//              long lda,
//              double *W,
//              long ldw,
//              double *R,
//              long ldr,
//              double *work,
//              int *info);

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
