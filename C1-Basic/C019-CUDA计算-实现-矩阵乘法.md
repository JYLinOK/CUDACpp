# C015-CUDA计算-实现-向量减法

## cu代码

代码：CudaRuntime10

```cpp
#include <stdio.h>
#include <stdio.h>  // 引入标准输入输出头文件
#include <chrono>   // 引入C11计时新库chrono
#include <iostream> 
#include <device_launch_parameters.h>  // 引入设备运行参数头文件
#include "device_functions.h"  // 引入相关设别函数：比如__syncthreads()
#include "cuda.h"  // 引入CUDA头文件
#include "cuda_runtime.h"  // 引入CUDA运行时头函数

using namespace std;  // 比如导入std才能使用命名空间chrono
using namespace chrono;

//threadIdx => 一个uint3类型，表示一个线程的索引。
//blockIdx => 一个uint3类型，表示一个线程块的索引。通常一个线程块中有多个线程。
//blockDim => 一个dim3类型，表示一个线程块的大小。
//gridDim => 一个dim3类型，表示一个网格的大小。通常一个网格中有多个线程块。


__global__ void vectorMatrixesAdd(const float* A, const float* B, float* C, int ARows, int AColumns, int BColumns) {
    // 矩阵A与矩阵B相乘的结果存于矩阵C
    // AColumns = BRows:矩阵A乘以矩阵B => 矩阵A的行数等于矩阵B的列数 
    // Row表示矩阵的行数，Column表示矩阵的列数
    // 使用共享内存来存储部分结果

    int column = blockIdx.x * blockDim.x + threadIdx.x;  // 获取二维-x维度索引数
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 获取二维-y维度索引数

    // 判断：如果逐行计算未完成且逐列计算也未完成
    if (row < ARows && column < BColumns) {
        // row行，column列的C元素的初始值
        float CValue = 0.0f;
        for (int r = 0; r < ARows; r++) {
            CValue += A[row * AColumns + r] * B[r * BColumns + column];
        }
        C[row * BColumns + column] = CValue;
    }
}


// 打印矩阵-相同维度
void printMatrixSameRC(float** FMatrix, int RC) {
    for (int i = 0; i < RC; i++) {
        for (int j = 0; j < RC; j++) {
            cout << *((float*)FMatrix + RC * i + j) << " ";
        }
        cout << endl;
    }
}

// 打印矩阵-不同维度
void printMatrixDifRC(float** FMatrix, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cout << *((float*)FMatrix + R * i + j) << " ";
        }
        cout << endl;
    }
}

void run(int Num) {
    // C++或者CUDA中的常规矩阵通过二维数组的数据结构表达
    // 设置矩阵的行数与列数
    // ARows, AColumns, BColumns
    const int ARows = 4; // A矩阵行数
    const int AColumns = 5; // A矩阵列数
    const int BRows = 5; // B矩阵行数
    const int BColumns = 6; // B矩阵列数
    float A[ARows][AColumns], B[BRows][BColumns], C[ARows][BColumns];

    // 为向量A B赋初始值
    for (int i = 0; i < ARows; i++) {
        for (int j = 0; j < AColumns; j++) {
            A[i][j] = static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < BRows; i++) {
        for (int j = 0; j < BColumns; j++) {
            B[i][j] = static_cast<float>(i + j);
        }
    }

    // 赋值之后下一步就是进行CUDA设备内存分配
    float* e_A, * e_B, * e_C;
    int Asize = ARows * AColumns * sizeof(float);
    int Bsize = BRows * BColumns * sizeof(float);
    int Csize = ARows * BColumns * sizeof(float);
    cudaMalloc((void**)&e_A, Asize);
    cudaMalloc((void**)&e_B, Bsize);
    cudaMalloc((void**)&e_C, Csize);

    // 计算执行前，将数据，从CUDA设备内存拷贝到主机内存
    cudaMemcpy(e_A, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(e_B, B, Bsize, cudaMemcpyHostToDevice);

    // 设置单块线程数
    dim3 threadsPerBlock(16, 16);
    // dim3是一个用于存储块和网格尺寸的结构类型
    // 设置单网格块数
    dim3 blocksPerGrid((BColumns + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用CUDA函数
    vectorMatrixesAdd <<< blocksPerGrid, threadsPerBlock >>> (e_A, e_B, e_C, ARows, AColumns, BColumns);

    // 计算完成后，将数据，从CUDA设备内存拷贝回主机内存
    cudaMemcpy(C, e_C, Csize, cudaMemcpyDeviceToHost);

    // 输出矩阵A与矩阵B
    printf("\n");
    cout << "矩阵A = " << endl;
    printMatrixDifRC((float**)A, ARows, AColumns);
    printf("\n");
    cout << "矩阵B = " << endl;
    printMatrixDifRC((float**)B, BRows, BColumns);

    // 输出结果矩阵 C
    printf("\n");
    cout << "矩阵A与矩阵B的相加矩阵 => C = " << endl;
    printMatrixDifRC((float**)C, ARows, BColumns);
    printf("\n");

    // 计算完成后，将CUDA设备内存清空，以备下次计算
    cudaFree(e_A);
    cudaFree(e_B);
    cudaFree(e_C);

    // 计算完成后，将主机变量内存清空，以备下次计算
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}


int main() {

    // ===========================================================
    auto start = system_clock::now();  // 获取当前时间-1
    // ============================================
    run(4);
    // ============================================
    auto end = system_clock::now();  // 获取当前时间-2
    // microseconds表示精确到微秒，nanoseconds表示精确到纳秒
    auto duration = duration_cast<microseconds>(end - start);  // 需要在函数内部使用duration_cast
    cout.setf(ios::fixed, ios::floatfield); // 设置使用十进制计数法，不使用科学计数法
    cout << "调用核函数时间：" << double(duration.count()) * microseconds::period::num / microseconds::period::den << "秒" << endl;
    // ===========================================================

    return 0;
}
```

## 输出：

```bash

矩阵A =
0 1 2 3 4
4 1 2 3 4
4 5 2 3 4
4 5 6 3 4

矩阵B =
0 1 2 3 4 5
5 1 2 3 4 5
5 6 2 3 4 5
5 6 7 3 4 5
5 6 7 8 4 5

矩阵A与矩阵B的相加矩阵 => C =
14 20 26 32 38 44
38 44 20 30 40 50
40 50 60 70 26 40
26 40 54 68 82 96

调用核函数时间：0.122083秒
```