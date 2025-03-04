
#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// #include "fileOpTool.h"

static __inline__ __device__ double warpAllReduceSum(double val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// __device__ volatile int com[8192] = {0};
// __device__ volatile int com[32768] = {0};
// __device__ volatile int com[65536] = {0};

// __device__ volatile int g_overFlag = 0;
__device__ int g_overFlag = 0;

// 输入参数：
// n表示对条带化称矩阵S的维度为nxn
// b表示其条带化的bandwidth为b

// 创建启动的线程数为<N,N>
// 一般情况下N=32,创建的是方阵个函数
// N = blockDim.x = blockDim.y
// BandWidth = b;这里只是为了函数加载时不报错

// 本函数实现如下三个功能:
// 1.把1个block中的warp分为两个部分,1部分进行数据预取,1部分进行数据的处理
// 2.利用对称性原理只处理矩阵的下三角部分
// 3.对B2部分进行循环利用
template <int BandWidth>
// __device__ void chasing_kernel_one_timeV7(int n, int b, double *subA, int ldSubA, int startRow)
__global__ void chasing_kernel_one_timeV10(int n,
                                           int b,
                                           double *subA,
                                           int ldSubA,
                                           double *dU,
                                           long ldU,
                                           int blockNum,
                                           int *com
                                          //  int *g_overFlag
                                          )
{
  auto grid  = cg::this_grid();
  // auto block = cg::this_thread_block();

  // 错误检查
  int Nx = blockDim.x;
  int Ny = blockDim.y;

  // 内部使用,不进行判断
  if (BandWidth != b)
  {
    return;
  }

  int warpGroupThdCount = Ny / 2;

  int bInx = grid.block_rank();

  int i = threadIdx.x;
  int j = threadIdx.y;

  __shared__ double u[BandWidth];

  __shared__ double S1[BandWidth * BandWidth]; // 共享内存,用于存放[B1,S,B2]进行变换
  __shared__ double S2[BandWidth * BandWidth]; // 共享内存,用于存放[B1,S,B2]进行变换
  int ldSS = b;

  double nu;
  double utx;

  for (; 0 == g_overFlag; bInx += blockNum)
  {
    int opColB1;
    int opRowB1;

    // 求出u的所在的列数
    double *uB = dU + bInx * ldU;

    long opRow = bInx + 1;

    int rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
    int colB1 = 1;

    // 找到B1在SubA中的位置
    // 所在列数为opRow-colB1,所在行数为colB1
    double *B1 = subA + colB1 + (opRow - colB1) * ldSubA;

    bool firstFlag = true;
    bool cycFlag   = true;

    // 最开始的时候S没有数据
    int rowS;
    int colS;

    double *S; // 初始的时候因为不会真正的使用它,所以不赋值没问题

    while (cycFlag && 0 == g_overFlag)
    {
      if ((bInx < n - 2) && (false == ((0 != bInx) && (opRow + 2 * b > com[bInx - 1]))))
      {
        if (true == firstFlag)
        {
#pragma unroll
          for (opColB1 = j; opColB1 < colB1; opColB1 += Ny)
          {
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              // 由于是阶梯状,所以B1每列会多减去列数
              S1[opRowB1 + opColB1 * ldSS] = B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
            }
          }

          // 最开始的时候S没有数据
          rowS = 0;
          colS = 0;
        }

        firstFlag = false;

        // 3.1 第1组warp: 对S2进行处理--从S2中写回preS,取新的A到S2中
        // 3.1 第2组warp(j==0): 对S1进行处理--取B1第1列到U中;求Householder向量
        if (0 != j)
        {
          // 第1组warp: 对S2进行处理--从S2写回preS,取新的A到S2中

          // 想把写回和读入编写到1个循环中,但是preS和S的形状大小可能不一样(特指第1和最后1个),所以分开实现

          // 注意在写回S的时候我们使用对称性,只写回去下三角部分的元素
#pragma unroll
          for (int opColS = j - 1; opColS < colS; opColS += (Ny - 1))
          {
#pragma unroll
            for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
            {
              // pB2[opRowB2][opColB2] = B1[opColB2][opRowB2] = SS[opColB2][opRowB2]
              // 由于是阶梯状,所以S每列会多减去列数
              S[(opRowS - opColS) + opColS * ldSubA] = S2[opRowS + opColS * ldSS];
            }
          }

          // 更新新的S
          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;
          // 读出新的S

          // if ((0 == i) && (1 == j))
          // {
          //     printf("block[%d] [%d][%d] come line=%d, opRow=%d, colS =%d.\n", bInx, i, j,
          //     __LINE__, opRow, colS); printf("\b");
          // }

          // 利用对称性,只拷贝下三角部分的元素
#pragma unroll
          for (int opColS = j - 1; opColS < colS; opColS += (Ny - 1))
          {
#pragma unroll
            for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
            {
              // pB2[opRowB2][opColB2] = B1[opColB2][opRowB2] = SS[opColB2][opRowB2]
              // 由于是阶梯状,所以S每列会多减去列数
              // S2[opColS + opRowS * ldSS] =
              S2[opRowS + opColS * ldSS] = S[(opRowS - opColS) + opColS * ldSubA];

              // 利用对称性
              S2[opColS + opRowS * ldSS] = S2[opRowS + opColS * ldSS];
            }
          }

          // if ((0 == i) && (1 == j))
          // {
          //     printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
          //     printf("\b");
          // }
        }
        else
        {
          // 第2组warp: 对S1进行处理--取B1第1列到U中;求Householder向量;对B1进行Householder变换
          // 2.1将B1中第一列的数据拷贝到u中
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            u[opRowB1] = S1[opRowB1];
          }

          __syncwarp();

          // 2.2 求出norm_x
          // double nu = 0.0;
          nu = 0.0;

#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            nu += u[opRowB1] * u[opRowB1];
          }

          // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
          double norm_x_squre = warpAllReduceSum(nu);
          double norm_x       = sqrt(norm_x_squre);

          // 2.3、求u=x/norm(x);
          double scale = 1.0 / norm_x;
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            u[opRowB1] *= scale;
          }

          __syncwarp();

          // 2.4、求u(0)= u(0)+sign(u(0)); 每列找一个线程来计算即可
          if (0 == i)
          {
            double u1 = u[0];

            u[0] += (u1 >= 0) ? 1 : -1;

            // 把normx存放到RR中，也就是对角线的元素
            // 使用这个值可以少进行一步计算,暂时没考虑,后期考虑
            // double RR = (u1 >= 0) ? -norm_x : norm_x;
          }

          __syncwarp();

          // 2.5、u=u/sqrt(abs(u(0))),计算HouseHolder向量
          scale = 1 / (sqrt(abs(u[0])));
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            u[opRowB1] *= scale;
          }

// 将求出的u向量放置到uB中
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            uB[opRow + opRowB1] = u[opRowB1];
          }

          // 更新新的S -- 这些warp也需要更新
          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;
        }

        __syncthreads();

        // 3.1.2 一起对B1进行Householder变换
#pragma unroll
        for (opColB1 = j; opColB1 < colB1; opColB1 += Ny)
        {
          nu = 0.0;
          // 先计算u'x
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            nu += u[opRowB1] * S1[opRowB1 + opColB1 * ldSS];
          }

          utx = warpAllReduceSum(nu);

          // 计算x-uu'x
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            S1[opRowB1 + opColB1 * ldSS] -= utx * u[opRowB1];
          }

          __syncwarp();
        }

        __syncthreads();

        // if ((0 == i) && (0 == j))
        // {
        //     printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
        //     printf("\b");
        // }

        // 3.2 第1组warp: 对S2进行处理--写回B1(包括B1和其转置位置),取B2到S1中
        // 注意: 取B2前都要判断数据同步条件--其实就是在B2变化的时候进行修改和判断同步条件
        // 3.2 第2组warp: 对S1进行处理--对S进行Householder变换
        if (j < warpGroupThdCount)
        {
          // 第1组warp: 对S2进行处理--写回B1(包括B1和其转置位置),取B2到S1中
#pragma unroll
          for (opColB1 = j; opColB1 < colB1; opColB1 += warpGroupThdCount)
          {
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              B1[(opRowB1 - opColB1) + opColB1 * ldSubA] = S1[opRowB1 + opColB1 * ldSS];

              // 写出B1转置
              // B1_T[opColB1 + opRowB1 * ldS] = S1[opRowB1 + opColB1 * ldSS];
            }
          }

          opRow += rowB1;                   // 更新下一次A的长度
          rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

          // 将同步条件写进去--不能写入同步条件,因为要保证数据一致性
          // 这儿不判断退出--也是因为数据同步,其他的warp可能还在处理

          // 取B2到S1中
#pragma unroll
          for (opColB1 = j; opColB1 < colB1; opColB1 += warpGroupThdCount)
          {
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              // SS[opRowB1][opColB1] = B1[opRowB1][opColB1] = B2[opColB1][opRowB1]
              S1[opRowB1 + opColB1 * ldSS] = B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
            }
          }

          // if ((0 == i) && (0 == j))
          // {
          //     printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
          //     printf("\b");
          // }
        }
        else
        {
          // 第2组warp: 对S1进行处理--对S进行Householder变换
#pragma unroll
          for (int opColS = j - warpGroupThdCount; opColS < colS; opColS += warpGroupThdCount)
          {
            nu = 0.0;
            // 先计算u'x
#pragma unroll
            for (int opRowS = i; opRowS < rowS; opRowS += Nx)
            {
              nu += u[opRowS] * S2[opRowS + opColS * ldSS];
            }

            utx = warpAllReduceSum(nu);

            // 计算x-uu'x
#pragma unroll
            for (int opRowS = i; opRowS < rowS; opRowS += Nx)
            {
              S2[opRowS + opColS * ldSS] -= utx * u[opRowS];
            }

            __syncwarp();
          }

          // 这里面的线程也需要更新这些局部变量
          opRow += rowB1;                   // 更新下一次A的长度
          rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

          // if ((0 == i) && (16 == j))
          // {
          //     printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
          //     printf("\b");
          // }
        }

        __syncthreads();
#pragma unroll
        for (int opRowS = j; opRowS < rowS; opRowS += Ny)
        {
          nu = 0.0;
          // 先计算u'x
#pragma unroll
          for (int opColS = i; opColS < colS; opColS += Nx)
          {
            nu += u[opColS] * S2[opRowS + opColS * ldSS];
          }

          utx = warpAllReduceSum(nu);

          // 计算x-uu'x
#pragma unroll
          for (int opColS = i; opColS < colS; opColS += Nx)
          {
            S2[opRowS + opColS * ldSS] -= utx * u[opColS];
          }

          __syncwarp();
        }

        __syncthreads();

#if MY_DEBUG
        if ((0 == i) && (0 == j))
        {
          printf("block[%d] [%d][%d] come line=%d, opRow=%d, rowB1 =%d, colB1=%d.\n",
                 bInx,
                 i,
                 j,
                 __LINE__,
                 opRow,
                 rowB1,
                 colB1);
          printf("\b");
        }
#endif

        // 3.3 两组warp一起对S1进行处理--对于B2进行Householder变换
#pragma unroll
        for (opRowB1 = j; opRowB1 < rowB1; opRowB1 += Ny)
        {
          nu = 0.0;
          // 先计算u'x
#pragma unroll
          for (opColB1 = i; opColB1 < colB1; opColB1 += Nx)
          {
            nu += u[opColB1] * S1[opRowB1 + opColB1 * ldSS];
          }

#if MY_DEBUG
          if ((0 == i) && (15 == j))
          {
            printf("block[%d] [%d][%d] come line=%d, opRow=%d, rowB1 =%d, colB1=%d,opRowB1 =%d, "
                   "opColB1=%d.\n",
                   bInx,
                   i,
                   j,
                   __LINE__,
                   opRow,
                   rowB1,
                   colB1,
                   opRowB1,
                   opColB1);
            printf("\b");
          }
#endif

          utx = warpAllReduceSum(nu);

          // 计算x-uu'x
#pragma unroll
          for (opColB1 = i; opColB1 < colB1; opColB1 += Nx)
          {
            S1[opRowB1 + opColB1 * ldSS] -= utx * u[opColB1];
          }

          // #if MY_DEBUG
          //             if ((0 == i) && (15 == j))
          //             {
          //                 printf("block[%d] [%d][%d] come line=%d, opRow=%d, rowB1 =%d,
          //                 colB1=%d,opRowB1 =%d, opColB1=%d .\n",
          //                        bInx, i, j, __LINE__, opRow, rowB1, colB1, opRowB1, opColB1);
          //                 printf("\b");
          //             }
          // #endif
          __syncwarp();
        }

        __syncthreads();

        // 判断退出条件
        if (rowB1 <= 1)
        {

#pragma unroll
          for (int opColS = j; opColS < colS; opColS += Ny)
          {
#pragma unroll
            for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
            {
              // 由于是阶梯状,所以S每列会多减去列数
              S[(opRowS - opColS) + opColS * ldSubA] = S2[opRowS + opColS * ldSS];
            }
          }

          // 写回B2
#pragma unroll
          for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny)
          {
#pragma unroll
            for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              // 写出B1和B1转置
              // 由于是阶梯状,所以B1每列会多减去列数
              B1[(opRowB1 - opColB1) + opColB1 * ldSubA] = S1[opRowB1 + opColB1 * ldSS];

              // 写出B1转置
              // B1_T[opColB1 + opRowB1 * ldS] = S1[opRowB1 + opColB1 * ldSS];
            }
          }

          __syncthreads();

          if ((n - 3) == bInx && 0 == threadIdx.x && 0 == threadIdx.y)
          {
            // if (0 == threadIdx.x && 0 == threadIdx.y)
            // {
            //   printf("[s3] bInx = %d, line = %d.\n", bInx, __LINE__);
            // }
            g_overFlag = 1;
          }
          __syncthreads();

          // break;

          cycFlag = false;
        }

        // 需要写入同步条件
        if ((0 == i) && (0 == j))
        {
          com[bInx] = opRow;

          if (false == cycFlag)
          {
            com[bInx] = n + 3 * b;
          }
        }

        __syncthreads();
      }

      grid.sync();
    }
  }
}
