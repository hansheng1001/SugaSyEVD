
// 根据条件数和特征值分布类型生成矩阵
// case 1:只有两个奇异值或特征值(1.0和条件数), 只有最小的4个数为1.0,其余数为条件数
// case 2: 只有两个奇异值或特征值(1.0和条件数), 只有最小的4个数为条件数,其余数为1.0
// case 3: 等比分布
// case 4: 等差分布
// case 5: 正态分布
// case 6: 均匀分布
void generateFloatMatrix(float *d_res_matrix, int m, int n, float start, float condition_number, int distribution_type);