#pragma once

#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// void generateUniformMatrix(double *dA, long int m, long int n);

// void generateUniformMatrixFloat(float *dA, long int m, long int n);

// void generateNormalMatrix(double *dA, long int m, long int n, double mean, double stddev);

// void generateNormalMatrixfloat(float *dA, long int m, long int n, float mean, float stddev);

// 定义函数模版
template <typename T>
void generateUniformMatrix(T *dA, long int m, long int n);

template <typename T>
void generateNormalMatrix(T *dA, long int m, long int n, T mean, T stddev);

void printDeviceMatrixV2Int(int *dA, long ldA, long rows, long cols);

std::vector<std::vector<double>> readMatrixFromFile(const std::string &fileName);

void fillMatrix(double *matrix, std::vector<std::vector<double>> &data);

void printMatrix(double *matrix, long ldA, long rows, long cols);

template <typename T>
void printDeviceMatrix(T *dA, long rows, long cols)
{
  T *matrix;
  matrix = (T *)malloc(sizeof(T) * rows * cols);

  cudaMemcpy(matrix, dA, sizeof(T) * rows * cols, cudaMemcpyDeviceToHost);

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {
      // printf("%f ", matrix[i * cols + j]);//按行存储优先
      printf("%10.4f", matrix[j * rows + i]); // 按列存储优先
    }
    printf("\n");
  }

  free(matrix);
}

template <typename T>
void printDeviceMatrixV2(T *dA, long ldA, long rows, long cols);

template <typename T>
void writeMatrixToCsvV2(T *dA, long ldA, long rows, long cols, const std::string &fileName)
{
  T matrix;

  std::ofstream file(fileName);

  if (file.is_open())
  {
    for (long i = 0; i < rows; i++)
    {
      for (long j = 0; j < cols; j++)
      {
        cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);
        file << matrix;
        if ((cols - 1) != j)
        {
          file << ",";
        }
      }
      file << std::endl;
    }
    file.close();
    std::cout << "Matrix written to " << fileName << std::endl;
  }
  else
  {
    std::cout << "Failed to open file: " << fileName << std::endl;
  }
}

template <typename T>
void printAndWriteMatrixToCsvV2(T *dA, long ldA, long rows, long cols, const std::string &fileName)
{
  T matrix;

  std::ofstream file(fileName);

  if (file.is_open())
  {
    for (long i = 0; i < rows; i++)
    {
      for (long j = 0; j < cols; j++)
      {
        cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);

        printf("%10.4f", matrix); // 按列存储优先

        file << matrix;
        if ((cols - 1) != j)
        {
          file << ",";
        }
      }
      printf("\n");
      file << std::endl;
    }
    file.close();
    std::cout << std::endl << "Matrix written to " << fileName << std::endl;
  }
  else
  {
    std::cout << "Failed to open file: " << fileName << std::endl;
  }
}

// 这是一个模板形式的把矩阵写入到文件中的一个函数
// template <typename T>
// void printMatrixDeviceBlock(char *filename, int m, int n, T *dA, int lda)
// {
//     FILE *f = fopen(filename, "w");
//     if (f == NULL)
//     {
//         printf("fault!\n");
//         return;
//     }
//     // printf("Perform printmatrixdevice\n");
//     float *ha;
//     ha = (float *)malloc(sizeof(float));

//     for (int i = 0; i < m; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             cudaMemcpy(&ha[0], &dA[i + j * lda], sizeof(float),
//                        cudaMemcpyDeviceToHost);
//             fprintf(f, "%lf", ha[0]);
//             if (j == n - 1)
//                 fprintf(f, "\n");
//             else
//                 fprintf(f, ",");
//         }
//     }
//     fclose(f);
//     // cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
//     // printMatrixFloat(filename, m, n, ha, lda);
//     free(ha);
// }