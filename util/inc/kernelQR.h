#include <cuda_fp16.h>

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <long M, long N>
__global__ void my_hou_kernel(long m, long n, half *A, long lda, half *R, long ldr)
{

  using T = half;

  // printf("come 19\n");
  // // cudaDeviceSynchronize();
  // __syncthreads();

  // 1.求出本block处理的矩阵的尺寸
  long mm = min(m - blockIdx.x * M, M);

  // 理论上不会出现mm<=0的情况，这里只是错误处理
  if (0 >= mm)
  {
    return;
  }

  A = A + blockIdx.x * M;
  R = R + blockIdx.x * N;

  // printf("come 28\n");
  // // cudaDeviceSynchronize();
  // __syncthreads();

  // 目前的情况只会出现n<=N的情况，目前只处理n=N=32的情况
  long nn = min(N, n);
  
  // 根据panelQR,目前处理的N一定等于N
  if(N != n)
  {
    return;
  }


  // 2. 找到本线程的ID
  // long i = blockIdx.x * blockDim.x + threadIdx.x;
  // long j = blockIdx.y * blockDim.y + threadIdx.y;
  long i = threadIdx.x;
  long j = threadIdx.y;

  // 创建shared memory，让整个block的线程能够进行数据共享
  // __shared__ double AA[mm * nn], RR[nn];
  __shared__ T AA[M * N], RR[N];
  long ldaa = mm;

  // 每个线程处理的数据个数
  // long rowDataNum = (mm + (blockDim.x - 1)) / blockDim.x;
  // long colDataNum = (nn + (blockDim.y - 1)) / blockDim.y;

  // double acc[rowDataNum];
  // T acc[8];

  // 将相应的A拷贝到共享内存当中
  for(long h = j; h < nn; h += blockDim.y)
  {
    for(long k = i; k < mm; k += blockDim.x)
    {
      AA[k + h * ldaa] = A[k + h * lda];
    }
  }

  // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
  __syncthreads();

  // 进行HouseHolder分解，先计算HouseHolder向量
  // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
  for (long cols = 0; cols < nn; cols++)
  {
    // 先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))

    if (j == cols % blockDim.y)
    {

      T nu = 0.0;
      // 0.求normx
      // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          nu += AA[k + cols*ldaa] * AA[k + cols*ldaa];
        }
      }

      // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
      T norm_x_squre = warpAllReduceSum(nu);
      T norm_x       = hsqrt(norm_x_squre);

      // 1、求u=x/norm(x);
      T scale = __float2half(1.0) / norm_x;
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          AA[k + cols*ldaa] *=scale;
        }
      }

      // __syncwarp();

      // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
      if (0 == i)
      {
        T u1 = AA[cols + cols * mm];

        AA[cols + cols * ldaa] += (u1 >= half(0)) ? 1 : -1;

        // 把normx存放到RR中，也就是对角线的元素
        RR[cols] = (u1 >= half(0)) ? -norm_x : norm_x;
      }

      __syncwarp();

      // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
      scale = half(1) / (hsqrt(__habs(AA[cols + cols * ldaa])));
#pragma unroll
      for(long k = i; k < mm; k += blockDim.x)
      {
        if(k >= cols)
        {
          AA[k + cols * ldaa] *= scale;
        }
      }
    }

    __syncthreads();
    // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
    // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
    // 每个线程按列需要处理多个列
    #pragma unroll
    for(long h = j; h < nn; h += blockDim.y)
    {
      // 只对大于cols的列进行HouseHolder变换
      if(h > cols)
      {
        T nu = 0.0;
        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            nu += AA[k + cols * ldaa] * AA[k + h * ldaa];
          }
        }
        T utx = warpAllReduceSum(nu);

        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            AA[k + h * ldaa] -= utx*AA[k + cols * ldaa];
          }
        }
        // __syncwarp();
      }
    }

  }

  __syncthreads();
  // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

  // 获得R矩阵，将AA的上三角部分拷贝到R中
  // 以R矩阵来进行循环
  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i; k < nn; k += blockDim.x)
    {
      if(k < h)
      {
        R[k + h*ldr] = AA[k + h*ldaa];
        AA[k + h*ldaa] = 0.0;
      }else if ( k > h)
      {
        R[k + h*ldr] = 0.0;
      }else
      {
        R[h + h*ldr] = RR[h];
      }
    }
  }

  // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
  // q表示是Q矩阵的1列

  // 注意:q=(I-u1*u1')(I-u2*u2')...(I-un*un')q;从后往前计算,注意由于最开始的q为
  // 单位向量Ik, 由于k行以下的元素为0,所以只有小于等于k的u才对q有贡献。

  // 每个线程最多处理的Q的元素数=Q最大行数/blockDim.x=mm/blockDim.x
  // 如果blockDim.x = 32,那么能够处理的Q的最大行数为8*32=256
  #define MAX_THREAD_PROC_Q_ELEMENTS 8
  T q[MAX_THREAD_PROC_Q_ELEMENTS];

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i, t = 0; k< mm; k += blockDim.x, ++t)
    {
      q[t] = 0.0;
      if(h == k)
      {
        q[t] = 1.0;
      }
    }
    __syncwarp();

    #pragma unroll
    for(long hh = h; hh >= 0; --hh)
    {
      T nu = 0.0;
      #pragma unroll
      for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
      {
        nu += q[t] * AA[k + hh*ldaa];
      }

      T utx = warpAllReduceSum(nu);

      #pragma unroll
      for(long k = i, t = 0; k < mm; k += blockDim.x, ++t)
      {
        q[t] -= utx * AA[k + hh*ldaa];
      }
      // __syncwarp();
    }

    #pragma unroll
    for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
    {
      A[k + h*lda] = q[t];
    }
    // __syncwarp();
  }

}


// 注意：目前此函数只处理n=N=32的情况
// 另外此函数只考虑mm>nn的情况，
// 同时注意此函数假设1个block的大小,即blockDim为<32,16>；<M,N>为<256,32>.
// 其实对于M != 256也是可以处理的

template <long M, long N>
__global__ void my_hou_kernel(long m, long n, double *A, long lda, double *R, long ldr)
{

  using T = double;

  // printf("come 19\n");
  // // cudaDeviceSynchronize();
  // __syncthreads();

  // 1.求出本block处理的矩阵的尺寸
  long mm = min(m - blockIdx.x * M, M);

  // 理论上不会出现mm<=0的情况，这里只是错误处理
  if (0 >= mm)
  {
    return;
  }

  A = A + blockIdx.x * M;
  R = R + blockIdx.x * N;

  // printf("come 28\n");
  // // cudaDeviceSynchronize();
  // __syncthreads();

  // 目前的情况只会出现n<=N的情况，目前只处理n=N=32的情况
  long nn = min(N, n);
  
  // 根据panelQR,目前处理的N一定等于N
  if(N != n)
  {
    return;
  }


  // 2. 找到本线程的ID
  // long i = blockIdx.x * blockDim.x + threadIdx.x;
  // long j = blockIdx.y * blockDim.y + threadIdx.y;
  long i = threadIdx.x;
  long j = threadIdx.y;

  // 创建shared memory，让整个block的线程能够进行数据共享
  // __shared__ double AA[mm * nn], RR[nn];
  __shared__ T AA[M * N], RR[N];
  long ldaa = mm;

  // 每个线程处理的数据个数
  // long rowDataNum = (mm + (blockDim.x - 1)) / blockDim.x;
  // long colDataNum = (nn + (blockDim.y - 1)) / blockDim.y;

  // double acc[rowDataNum];
  // T acc[8];

  // 将相应的A拷贝到共享内存当中
  for(long h = j; h < nn; h += blockDim.y)
  {
    for(long k = i; k < mm; k += blockDim.x)
    {
      AA[k + h * ldaa] = A[k + h * lda];
    }
  }

  // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
  __syncthreads();

  // 进行HouseHolder分解，先计算HouseHolder向量
  // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
  for (long cols = 0; cols < nn; cols++)
  {
    // 先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))

    if (j == cols % blockDim.y)
    {

      T nu = 0.0;
      // 0.求normx
      // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          nu += AA[k + cols*ldaa] * AA[k + cols*ldaa];
        }
      }

      // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
      T norm_x_squre = warpAllReduceSum(nu);
      T norm_x       = sqrt(norm_x_squre);

      // 1、求u=x/norm(x);
      T scale = 1.0 / norm_x;
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          AA[k + cols*ldaa] *=scale;
        }
      }

      // __syncwarp();

      // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
      if (0 == i)
      {
        T u1 = AA[cols + cols * mm];

        AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;

        // 把normx存放到RR中，也就是对角线的元素
        RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
      }

      __syncwarp();

      // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
      scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));
#pragma unroll
      for(long k = i; k < mm; k += blockDim.x)
      {
        if(k >= cols)
        {
          AA[k + cols * ldaa] *= scale;
        }
      }
    }

    __syncthreads();
    // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
    // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
    // 每个线程按列需要处理多个列
    #pragma unroll
    for(long h = j; h < nn; h += blockDim.y)
    {
      // 只对大于cols的列进行HouseHolder变换
      if(h > cols)
      {
        T nu = 0.0;
        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            nu += AA[k + cols * ldaa] * AA[k + h * ldaa];
          }
        }
        T utx = warpAllReduceSum(nu);

        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            AA[k + h * ldaa] -= utx*AA[k + cols * ldaa];
          }
        }
        // __syncwarp();
      }
    }

  }

  __syncthreads();
  // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

  // 获得R矩阵，将AA的上三角部分拷贝到R中
  // 以R矩阵来进行循环
  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i; k < nn; k += blockDim.x)
    {
      if(k < h)
      {
        R[k + h*ldr] = AA[k + h*ldaa];
        AA[k + h*ldaa] = 0.0;
      }else if ( k > h)
      {
        R[k + h*ldr] = 0.0;
      }else
      {
        R[h + h*ldr] = RR[h];
      }
    }
  }

  // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
  // q表示是Q矩阵的1列

  // 注意:q=(I-u1*u1')(I-u2*u2')...(I-un*un')q;从后往前计算,注意由于最开始的q为
  // 单位向量Ik, 由于k行以下的元素为0,所以只有小于等于k的u才对q有贡献。

  // 每个线程最多处理的Q的元素数=Q最大行数/blockDim.x=mm/blockDim.x
  // 如果blockDim.x = 32,那么能够处理的Q的最大行数为8*32=256
  #define MAX_THREAD_PROC_Q_ELEMENTS 8
  T q[MAX_THREAD_PROC_Q_ELEMENTS];

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i, t = 0; k< mm; k += blockDim.x, ++t)
    {
      q[t] = 0.0;
      if(h == k)
      {
        q[t] = 1.0;
      }
    }
    __syncwarp();

    #pragma unroll
    for(long hh = h; hh >= 0; --hh)
    {
      T nu = 0.0;
      #pragma unroll
      for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
      {
        nu += q[t] * AA[k + hh*ldaa];
      }

      T utx = warpAllReduceSum(nu);

      #pragma unroll
      for(long k = i, t = 0; k < mm; k += blockDim.x, ++t)
      {
        q[t] -= utx * AA[k + hh*ldaa];
      }
      // __syncwarp();
    }

    #pragma unroll
    for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
    {
      A[k + h*lda] = q[t];
    }
    // __syncwarp();
  }

}

template <long M, long N>
__global__ void my_hou_kernel(long m, long n, float *A, long lda, float *R, long ldr)
{

  using T = float;

  // printf("come 19\n");
  // // cudaDeviceSynchronize();
  // __syncthreads();

  // 1.求出本block处理的矩阵的尺寸
  long mm = min(m - blockIdx.x * M, M);

  // 理论上不会出现mm<=0的情况，这里只是错误处理
  if (0 >= mm)
  {
    return;
  }

  A = A + blockIdx.x * M;
  R = R + blockIdx.x * N;

  // printf("come 28\n");
  // // cudaDeviceSynchronize();
  // __syncthreads();

  // 目前的情况只会出现n<=N的情况，目前只处理n=N=32的情况
  long nn = min(N, n);
  
  // 根据panelQR,目前处理的N一定等于N
  if(N != n)
  {
    return;
  }


  // 2. 找到本线程的ID
  // long i = blockIdx.x * blockDim.x + threadIdx.x;
  // long j = blockIdx.y * blockDim.y + threadIdx.y;
  long i = threadIdx.x;
  long j = threadIdx.y;

  // 创建shared memory，让整个block的线程能够进行数据共享
  // __shared__ double AA[mm * nn], RR[nn];
  __shared__ T AA[M * N], RR[N];
  long ldaa = mm;

  // 每个线程处理的数据个数
  // long rowDataNum = (mm + (blockDim.x - 1)) / blockDim.x;
  // long colDataNum = (nn + (blockDim.y - 1)) / blockDim.y;

  // double acc[rowDataNum];
  // T acc[8];

  // 将相应的A拷贝到共享内存当中
  for(long h = j; h < nn; h += blockDim.y)
  {
    for(long k = i; k < mm; k += blockDim.x)
    {
      AA[k + h * ldaa] = A[k + h * lda];
    }
  }

  // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
  __syncthreads();

  // 进行HouseHolder分解，先计算HouseHolder向量
  // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
  for (long cols = 0; cols < nn; cols++)
  {
    // 先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))

    if (j == cols % blockDim.y)
    {

      T nu = 0.0;
      // 0.求normx
      // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          nu += AA[k + cols*ldaa] * AA[k + cols*ldaa];
        }
      }

      // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
      T norm_x_squre = warpAllReduceSum(nu);
      T norm_x       = sqrt(norm_x_squre);

      // 1、求u=x/norm(x);
      T scale = 1.0 / norm_x;
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          AA[k + cols*ldaa] *=scale;
        }
      }

      // __syncwarp();

      // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
      if (0 == i)
      {
        T u1 = AA[cols + cols * mm];

        AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;

        // 把normx存放到RR中，也就是对角线的元素
        RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
      }

      __syncwarp();

      // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
      scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));
#pragma unroll
      for(long k = i; k < mm; k += blockDim.x)
      {
        if(k >= cols)
        {
          AA[k + cols * ldaa] *= scale;
        }
      }
    }

    __syncthreads();
    // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
    // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
    // 每个线程按列需要处理多个列
    #pragma unroll
    for(long h = j; h < nn; h += blockDim.y)
    {
      // 只对大于cols的列进行HouseHolder变换
      if(h > cols)
      {
        T nu = 0.0;
        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            nu += AA[k + cols * ldaa] * AA[k + h * ldaa];
          }
        }
        T utx = warpAllReduceSum(nu);

        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            AA[k + h * ldaa] -= utx*AA[k + cols * ldaa];
          }
        }
        // __syncwarp();
      }
    }

  }

  __syncthreads();
  // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

  // 获得R矩阵，将AA的上三角部分拷贝到R中
  // 以R矩阵来进行循环
  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i; k < nn; k += blockDim.x)
    {
      if(k < h)
      {
        R[k + h*ldr] = AA[k + h*ldaa];
        AA[k + h*ldaa] = 0.0;
      }else if ( k > h)
      {
        R[k + h*ldr] = 0.0;
      }else
      {
        R[h + h*ldr] = RR[h];
      }
    }
  }

  // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
  // q表示是Q矩阵的1列

  // 注意:q=(I-u1*u1')(I-u2*u2')...(I-un*un')q;从后往前计算,注意由于最开始的q为
  // 单位向量Ik, 由于k行以下的元素为0,所以只有小于等于k的u才对q有贡献。

  // 每个线程最多处理的Q的元素数=Q最大行数/blockDim.x=mm/blockDim.x
  // 如果blockDim.x = 32,那么能够处理的Q的最大行数为8*32=256
  #define MAX_THREAD_PROC_Q_ELEMENTS 8
  T q[MAX_THREAD_PROC_Q_ELEMENTS];

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i, t = 0; k< mm; k += blockDim.x, ++t)
    {
      q[t] = 0.0;
      if(h == k)
      {
        q[t] = 1.0;
      }
    }
    __syncwarp();

    #pragma unroll
    for(long hh = h; hh >= 0; --hh)
    {
      T nu = 0.0;
      #pragma unroll
      for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
      {
        nu += q[t] * AA[k + hh*ldaa];
      }

      T utx = warpAllReduceSum(nu);

      #pragma unroll
      for(long k = i, t = 0; k < mm; k += blockDim.x, ++t)
      {
        q[t] -= utx * AA[k + hh*ldaa];
      }
      // __syncwarp();
    }

    #pragma unroll
    for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
    {
      A[k + h*lda] = q[t];
    }
    // __syncwarp();
  }

}
