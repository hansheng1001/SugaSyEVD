// #include < iostream>
#include <iostream>
#include <fstream>
#include <cusolverDn.h>
#include <vector>
#include <iomanip> // 包含操纵器的头文件

#include <curand.h>

#include "myBase.h"

__global__ void fillArrayCluster(float *arr, int size, float start, float condition, bool large)
{
    float largeSV = start * condition;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        arr[idx] = large ? largeSV : start;
        if (idx < 4)
            arr[idx] = large ? start : largeSV;
    }
}

// 对角线元素为等比分布--也就是奇异值(或者特征值)为等比分布
__global__ void geometricSpace(float *arr, int size, float start, float condition)
{
    float end = start * condition;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        float factor = powf(end / start, 1.0f / (size - 1));
        arr[idx] = start * powf(factor, idx);
    }
}

// 对角线元素为等差分布--也就是奇异值(或者特征值)为等差分布
__global__ void arithmeticSpace(float *arr, int size, float start, float condition)
{
    float end = start * condition;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        float step = (end - start) / (float)(size - 1);
        arr[idx] = start + step * idx;
    }
}

__global__ void setRandomValues(float *arr, int size, int type, float *rand_vals)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        arr[idx] = fabsf(rand_vals[idx]);
    }
}

__global__ void setEye(int m, int n, float *a, int lda)
{
    long int i = threadIdx.x + blockDim.x * blockIdx.x;
    long int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < m && j < n)
    {
        if (i == j)
            a[i + j * lda] = 1.0f;
        else
            a[i + j * lda] = 0.0f;
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkOrthogonal(int m, int n, float *A, int lda, int lde)
{
    float *eye_matrix;
    float *ATA;
    float d_norm_res;
    float snegone = -1.0f;
    float sone = 1.0f;
    float szero = 0.0f;
    int incx = 1;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // std::ofstream myfile;
    checkCudaError(cudaMalloc(&ATA, m * n * sizeof(float)), "cudaMalloc");
    // 生成单位矩阵
    checkCudaError(cudaMalloc(&eye_matrix, m * n * sizeof(float)), "cudaMalloc");
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    setEye<<<blocksPerGrid, threadsPerBlock>>>(m, n, eye_matrix, m);
    cudaDeviceSynchronize();
    // 测试dnrm2函数
    cublasSnrm2(handle, m * n, eye_matrix, incx, &d_norm_res);
    std::cout << "Eye Matrix Norm Result: " << d_norm_res << std::endl;

    // 测试单位矩阵的创建
    std::vector<float> h_eye_matrix(m * n);
    cudaMemcpy(h_eye_matrix.data(), eye_matrix, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // 测试正交性
    // 求出A的转置
    cublasSgeam(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n,
                &sone,
                A, m,
                &szero,
                A, m,
                ATA, n);

    // 计算I-A*A^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, m, &snegone, A, m, ATA, n, &sone, eye_matrix, m);
    // cudaMemcpy(h_eye_matrix.data(), eye_matrix, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    //  myfile.open("I-AAmatrix.txt");
    //  for (int i = 0; i < m; ++i) {
    //      for (int j = 0; j < n; ++j) {
    //          myfile << h_eye_matrix[i * n + j] << " ";
    //      }
    //      myfile << "\n";
    //  }
    //  myfile.close();

    cublasSnrm2(handle, m * n, eye_matrix, 1, &d_norm_res);
    std::cout << std::scientific;
    std::cout << std::setprecision(6);
    std::cout << "Norm Result: " << d_norm_res / m << std::endl;
}

void generateOrthogonalMatrix(float *d_random_matrix, int m, int n, unsigned long long seed)
{
    /*生成正交矩阵*/
    float done = 1.0f;
    float dzero = 0.0f;
    float dnegone = -1.0f;

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // 创建两个随机矩阵
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, d_random_matrix, m * n, 0.0f, 1.0f);
    curandDestroyGenerator(gen);

    int *d_info;
    cudaMalloc(&d_info, sizeof(int));
    int lwork = 0;
    float *d_work = NULL;
    // 计算工作空间
    cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_random_matrix, m, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));

    float *tau;
    cudaMalloc(&tau, m * sizeof(float));

    // 执行qr分解
    cusolverDnSgeqrf(cusolverH, m, n, d_random_matrix, m, tau, d_work, lwork, d_info);
    cudaMemcpy(&lwork, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (lwork != 0)
    {
        std::cerr << "QR decomposition1 failed" << std::endl;
        return;
    }

    // 生成正交矩阵
    cusolverDnSorgqr(cusolverH, m, n, n, d_random_matrix, m, tau, d_work, lwork, d_info);
    cudaMemcpy(&lwork, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (lwork != 0)
    {
        std::cerr << "Generating orthogonal matrix failed" << std::endl;
        return;
    }

    /*测试正交矩阵正交性*/
    checkOrthogonal(m, n, d_random_matrix, m, n);

    // Freeing allocated memory
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(tau);
    // cudaFree(eye_matrix);
}

// 根据条件数和特征值分布类型生成矩阵
// case 1:只有两个奇异值或特征值(start 和 start*条件数), 只有最小的4个数为1.0,其余数为条件数
// case 2: 只有两个奇异值或特征值(start和 start*条件数), 只有最小的4个数为条件数,其余数为1.0
// case 3: 等比分布
// case 4: 等差分布
// case 5: 正态分布
// case 6: 均匀分布
void generateFloatMatrix(float *d_res_matrix, int m, int n, float start, float condition_number, int distribution_type)
{

    // 这里默认m = n，不然需要调整代码
    int size = m;
    const char *output_file = "eigenvalues.txt";
    float alpha = 1.0f;
    float beta = 0.0f;
    float nbeta = -1.0f;
    float *d_diag;
    std::ofstream myfile;
    cudaMalloc(&d_diag, size * sizeof(float));

    curandGenerator_t gen;
    float *d_rand_vals;
    cudaMalloc(&d_rand_vals, size * sizeof(float));

    dim3 gridDim((size + 256 - 1) / 256);
    dim3 blockDim(256);

    switch (distribution_type)
    {
    case 1: // 只有两个奇异值或特征值(1.0和条件数), 只有最小的4个数为1.0,其余数为条件数
        fillArrayCluster<<<gridDim, blockDim>>>(d_diag, size, start, condition_number, true);
        break;
    case 2: // 只有两个奇异值或特征值(1.0和条件数), 只有最小的4个数为条件数,其余数为1.0
        fillArrayCluster<<<gridDim, blockDim>>>(d_diag, size, start, condition_number, false);
        break;
    case 3: // 等比分布
        geometricSpace<<<gridDim, blockDim>>>(d_diag, size, start, condition_number);
        break;
    case 4: // 等差分布
        arithmeticSpace<<<gridDim, blockDim>>>(d_diag, size, start, condition_number);
        break;
    case 5: // 正泰分布
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateNormal(gen, d_rand_vals, size * sizeof(float), 0.0f, 1.0f);
        setRandomValues<<<gridDim, blockDim>>>(d_diag, size, distribution_type, d_rand_vals);
        curandDestroyGenerator(gen);
        break;
    case 6: // 均匀分布
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_rand_vals, size * sizeof(float));
        setRandomValues<<<gridDim, blockDim>>>(d_diag, size, distribution_type, d_rand_vals);
        curandDestroyGenerator(gen);
        break;
    default:
        std::cerr << "Invalid distribution type" << std::endl;
        return;
    }

    cudaFree(d_rand_vals);

    // Error checking function definitions are assumed implemented elsewhere
    // checkCUDAError("Kernel execution failed");

    // print_to_file(d_diag, size, output_file);

    float *d_random_matrix, *d_random_matrix1;
    cudaMalloc(&d_random_matrix, size * size * sizeof(float));
    cudaMalloc(&d_random_matrix1, size * size * sizeof(float));
    generateOrthogonalMatrix(d_random_matrix, size, size, 1234ULL);
    generateOrthogonalMatrix(d_random_matrix1, size, size, 5678ULL);

    // 生成Q1 * diag * Q2‘矩阵
    // float* d_res_matrix;
    float *work;
    cudaMalloc(&work, size * size * sizeof(float));
    // cudaMalloc(&d_res_matrix, size * size * sizeof(float));
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // setEye<<<blocksPerGrid, threadsPerBlock>>>(size, size, work, size);
    // cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, size, size, d_random_matrix, size, d_diag, 1, work, size);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                &alpha, work, size, d_random_matrix1, size, &beta, d_res_matrix, size);

    cudaFree(d_diag);
    cudaFree(work);
    cudaFree(d_random_matrix);
    cudaFree(d_random_matrix1);
}

// 调用cublasSnrm2函数计算一个单精度矩阵的二范数，即矩阵中所有元素平方和开根号的值
float snorm(int m, int n, float *dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sn;
    int incx = 1;
    cublasSnrm2(handle, m * n, dA, incx, &sn);
    cublasDestroy(handle);
    return sn;
}

void checkOtho(int m, int n, float *Q, int ldq)
{
    float *I;
    cudaMalloc(&I, sizeof(float) * n * n);

    // printMatrixDeviceBlock("Q.csv",m,n,Q,m);

    dim3 grid96((n + 1) / 32, (n + 1) / 32);
    dim3 block96(32, 32);
    setEye<<<grid96, block96>>>(n, n, I, n);
    float snegone = -1.0;
    float sone = 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                 &snegone, Q, CUDA_R_32F, ldq, Q, CUDA_R_32F, ldq,
                 &sone, I, CUDA_R_32F, n, CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT);

    float normRes = snorm(n, n, I);
    printf("||I-Q'*Q||/N = %.6e\n", normRes / n);
    cudaFree(I);
    cublasDestroy(handle);
}

void sgemm(int m, int n, int k, float *dA, int lda, float *dB, int ldb, float *dC, int ldc, float alpha, float beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sone = alpha;
    float szero = beta;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &sone, dA, lda,
                dB, ldb,
                &szero, dC, ldc);
    cublasDestroy(handle);
}

void checkResult2(int m, int n, float *A, int lda, float *Q, int ldq, float *R, int ldr)
{
    float normA = snorm(m, n, A);
    float alpha = 1.0;
    float beta = -1.0;
    startTimer();
    sgemm(m, n, n, Q, ldq, R, ldr, A, lda, alpha, beta);
    float ms = stopTimer();
    printf("SGEMM m*n*k %d*%d*%d takes %.0f (ms), exec rate %.0f GFLOPS\n",
           m, n, n, ms, 2.0 * m * n * n / (ms * 1e6));
    float normRes = snorm(m, n, A);
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n", normRes / normA);
}