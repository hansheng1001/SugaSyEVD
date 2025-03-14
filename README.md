# HygonSyEVD: 目前只有SBR调试成功
# author: WangHansheng@UESTC
# email: wanghansheng@std.uestc.edu.cn

## Usage
```bash
cmake -B build -S .
cmake --build build/ -j
cd build/src/EVD
./my_EVD_useLAPACKE_CompQ [n] [b] [nb]

n: 对称矩阵的矩阵尺寸,目前仅支持2的幂次方;
b: 二阶段SBR中条带化矩阵的带宽,2的幂次方,小于等于n;
nb: 二阶段SBR中进行双块SBR的积攒块大小,必须是b的倍数且小于等于n。

备注: 可以通过打印的正交性误差和后向误差判断正确性。
