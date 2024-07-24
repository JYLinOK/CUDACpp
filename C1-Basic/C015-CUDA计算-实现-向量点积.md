# C015-CUDA计算-实现-向量点积

## cu代码

代码：CudaRuntime5

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


__global__ void vectorDotProduct(const float* A, const float* B, float* C, int N) {
    // 使用共享内存来存储部分结果
    //__shared__ float temp[256]; // 假设每个块最多有 256 个线程
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // 获取一维索引数=>计算全局线程索引
    
    // 每个线程计算部分点积=>最终用于全局结果
    float temp = 0.0f;
    if (index < N) {
        temp = A[index] * B[index];
    }

    // 使用原子操作将部分结果加到全局结果中
    atomicAdd(C, temp);
}


void run(int Num) {
    // 设置分配内存
    // void *malloc(size_t size)是C语言的库函数 
    // 用于分配指定的内存空间，并返回一个指向该空间的指针
    size_t size = Num * sizeof(float);
    float* mal_A = (float*)malloc(size);
    float* mal_B = (float*)malloc(size);
    float* mal_C = (float*)malloc(size);

    // 为向量A B赋初始值
    for (int i = 0; i < Num; i++) {
        mal_A[i] = i+1;
        mal_B[i] = (i+1) * 2.0f;
    }

    // 赋值之后下一步就是进行CUDA设备内存分配
    float* e_A, * e_B, * e_C;
    cudaMalloc(&e_A, size);
    cudaMalloc(&e_B, size);
    cudaMalloc(&e_C, size);

    // 计算执行前，将数据，从CUDA设备内存拷贝到主机内存
    cudaMemcpy(e_A, mal_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(e_B, mal_B, size, cudaMemcpyHostToDevice);

    // 设置单块线程数
    int threadsPerBlock = 128;
    // 设置单格块数
    int blocksPerGrid = (Num + threadsPerBlock - 1) / threadsPerBlock;

    // 调用CUDA函数
    vectorDotProduct <<< blocksPerGrid, threadsPerBlock >>> (e_A, e_B, e_C, Num);

    // 计算完成后，将数据，从CUDA设备内存拷贝回主机内存
    cudaMemcpy(mal_C, e_C, size, cudaMemcpyDeviceToHost);

    // 输出计算结果
    for (int i = 0; i < Num; i++) {
        printf(" mal_A[%d] = %f ", i, mal_A[i]);
    }

    printf("\n\n");

    for (int i = 0; i < Num; i++) {
        printf(" mal_B[%d] = %f ", i, mal_B[i]);
    }

    printf("\n\n");

    for (int i = 0; i < Num; i++) {
        printf(" mal_C[%d] = %f ", i, mal_C[i]);
    }

    printf("\n\n");

    printf("向量A与向量B的点积= %f \n\n", mal_C[0]);


    // 计算完成后，将CUDA设备内存清空，以备下次计算
    cudaFree(e_A);
    cudaFree(e_B);
    cudaFree(e_C);
    // 计算完成后，将主机内存清空，以备下次计算
    free(mal_A);
    free(mal_B);
    free(mal_C);
}

int main() {

    // ===========================================================
    auto start = system_clock::now();  // 获取当前时间-1
    // ============================================
    run(3);
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
 mal_A[0] = 1.000000  mal_A[1] = 2.000000  mal_A[2] = 3.000000

 mal_B[0] = 2.000000  mal_B[1] = 4.000000  mal_B[2] = 6.000000

 mal_C[0] = 28.000000  mal_C[1] = 0.000000  mal_C[2] = 0.000000

向量A与向量B的点积= 28.000000

调用核函数时间：1.879536秒

E:\CUDACppCodes\C1-Basic\CudaRuntime7\x64\Debug\CudaRuntime7.exe (进程 17288)已退出，代码为 0。
要在调试停止时自动关闭控制台，请启用“工具”->“选项”->“调试”->“调试停止时自动关闭控制台”。
按任意键关闭此窗口. . .
```