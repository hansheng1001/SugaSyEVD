
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

        if (0 != j)
        {
          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;

        }
        else
        {
          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;
        }

        __syncthreads();

        if (j < warpGroupThdCount)
        {
          opRow += rowB1;                   // 更新下一次A的长度
          rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

        }
        else
        {
          // 这里面的线程也需要更新这些局部变量
          opRow += rowB1;                   // 更新下一次A的长度
          rowB1 = min(b, (int)(n - opRow)); // 因为opRow是起始位置,所以是n-1 - opRow +1 = n - opRow;
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

        }

        // 判断退出条件
        if (rowB1 <= 1)
        {

          if ((n - 3) == bInx && 0 == threadIdx.x && 0 == threadIdx.y)
          {

            g_overFlag = 1;
          }
          __syncthreads();


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
