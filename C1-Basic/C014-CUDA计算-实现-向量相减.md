# C014-CUDA计算-实现-向量相减

## cu代码

代码：CudaRuntime5

```cpp
#include <stdio.h>
#include "cuda_runtime.h"  // 引入运行时头文件
#include "device_launch_parameters.h"  // 引入设备参数头文件
#include <stdio.h>  // 引入标准输入输出头文件
#include <chrono>   // 引入C11计时新库chrono
#include <iostream> 
#include <device_launch_parameters.h>  // 引入设备运行参数头文件

using namespace std;  // 比如导入std才能使用命名空间chrono
using namespace chrono;

/// <summary>
/// vectorAdd用于实现向量的加法
/// </summary>
/// <param name="A">两个相减向量中的一个</param>
/// <param name="B">两个相减向量中的另一个</param>
/// <param name="C">两个相减向量中的和向量-结果向量</param>
/// <param name="Num">向量的维度</param>
/// <returns></returns>
__global__ void vectorSubtract(const float* A, const float* B, float* C, int Num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Num) {
        C[i] = A[i] - B[i];
    }
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
        mal_A[i] = i;
        mal_B[i] = i * 1.0f;
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
    vectorAdd <<<blocksPerGrid, threadsPerBlock >>> (e_A, e_B, e_C, Num);

    // 计算完成后，将数据，从CUDA设备内存拷贝回主机内存
    cudaMemcpy(mal_C, e_C, size, cudaMemcpyDeviceToHost);

    // 输出计算结果
    for (int i = 0; i < Num; i++) {
        printf(" mal_C[%d] = %f ", i, mal_C[i]);
    }

    printf("\n");

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
    run(10);
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

输出：

```bash
 mal_C[0] = 0.000000  mal_C[1] = 0.000000  mal_C[2] = 0.000000  mal_C[3] = 0.000000  mal_C[4] = 0.000000  mal_C[5] = 0.000000  mal_C[6] = 0.000000  mal_C[7] = 0.000000  mal_C[8] = 0.000000  mal_C[9] = 0.000000
调用核函数时间：1.704989秒

E:\CUDACppCodes\C1-Basic\CudaRuntime6\x64\Debug\CudaRuntime6.exe (进程 25548)已退出，代码为 0。
要在调试停止时自动关闭控制台，请启用“工具”->“选项”->“调试”->“调试停止时自动关闭控制台”。
按任意键关闭此窗口. . .

```