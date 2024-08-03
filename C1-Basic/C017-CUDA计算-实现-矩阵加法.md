# C015-CUDA计算-实现-向量加法

## cu代码

代码：[../C0-Code/C1-Basic/CudaRuntime8/](../C0-Code/C1-Basic//CudaRuntime8/)

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


__global__ void vectorMatrixesAdd(const float* A, const float* B, float* C, int Rows, int Columns) {
    // 矩阵A与矩阵B相乘的结果存于矩阵C
    // Row表示矩阵的行数，Column表示矩阵的列数
    // 使用共享内存来存储部分结果

    int column = blockIdx.x * blockDim.x + threadIdx.x;  // 获取二维-x维度索引数
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 获取二维-y维度索引数

    // 逐行逐列计算元素相加
    if (row < Rows && column < Columns) {
        // 定义按行逐个排序时的元素索引
        int i = row * Columns + column;
        C[i] = A[i] + B[i];
    }
}

void printMatrix(float** FMatrix, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            cout << *((float*)FMatrix + dim * i + j) << " ";
        }
        cout << endl;
    }
}

void run(int Num) {
    // C++或者CUDA中的常规矩阵通过二维数组的数据结构表达
    // 设置矩阵的行数与列数
    const int rows = 4; // 矩阵行数
    const int columns = 4; // 矩阵列数
    float A[rows][columns], B[rows][columns], C[rows][columns];

    // 为向量A B赋初始值
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            A[i][j] = static_cast<float>(i + j);
            B[i][j] = static_cast<float>(i + j);
        }
    }

    // 赋值之后下一步就是进行CUDA设备内存分配
    float* e_A, * e_B, * e_C;
    int size = rows * columns * sizeof(float);
    cudaMalloc((void**)&e_A, size);
    cudaMalloc((void**)&e_B, size);
    cudaMalloc((void**)&e_C, size);

    // 计算执行前，将数据，从CUDA设备内存拷贝到主机内存
    cudaMemcpy(e_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(e_B, B, size, cudaMemcpyHostToDevice);

    // 设置单块线程数
    dim3 threadsPerBlock(16, 16);
    // dim3是一个用于存储块和网格尺寸的结构类型
    // 设置单网格块数
    dim3 blocksPerGrid((columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用CUDA函数
    vectorMatrixesAdd <<< blocksPerGrid, threadsPerBlock >>> (e_A, e_B, e_C, rows, columns);

    // 计算完成后，将数据，从CUDA设备内存拷贝回主机内存
    cudaMemcpy(C, e_C, size, cudaMemcpyDeviceToHost);

    // 输出矩阵A与矩阵B
    printf("\n");
    cout << "矩阵A = " << endl;
    printMatrix((float**)A, Num);
    printf("\n");
    cout << "矩阵B = " << endl;
    printMatrix((float**)B, Num);

    // 输出结果矩阵 C
    printf("\n");
    cout << "矩阵A与矩阵B的相加矩阵 = C = " << endl;
    printMatrix((float**)C, Num);
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
0 1 2 3
1 2 3 4
2 3 4 5
3 4 5 6

矩阵B =
0 1 2 3
1 2 3 4
2 3 4 5
3 4 5 6

矩阵A与矩阵B的相加矩阵 = C =
0 2 4 6
2 4 6 8
4 6 8 10
6 8 10 12

调用核函数时间：0.163650秒
```