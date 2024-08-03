#include "cuda_runtime.h"  // 引入运行时头文件
#include "device_launch_parameters.h"  // 引入设备参数头文件
#include <stdio.h>  // 引入标准输入输出头文件
#include <chrono>   // 引入C11计时新库chrono
#include <iostream> 

using namespace std;  // 比如导入std才能使用命名空间chrono
using namespace chrono;


/// <summary>
/// 定义核函数
/// </summary>
/// <param name="n">块数</param>
/// <param name="m">线程数</param>
/// <returns></returns>
__global__ void kernelFun(int n, int m) {
	printf("执行核函数内部, 块数目= %d, 线程数目= %d\n", n, m);
}


/// <summary>
/// 通过n个块m个线程启动kernel函数
/// </summary>
/// <param name="n">block块数目</param>
/// <param name="m">thread线程数目</param>
void runKernel(int n, int m) {
	cout << "块数目= " << n << ", 线程数目= " << m << endl;
	// ===========================================================
	auto start = system_clock::now();  // 获取当前时间-1
	// ============================================
	kernelFun <<<n, m>>> (n, m);
	// ============================================
	auto end = system_clock::now();  // 获取当前时间-2
	// microseconds表示精确到微秒，nanoseconds表示精确到纳秒
	auto duration = duration_cast<microseconds>(end - start);  // 需要在函数内部使用duration_cast
	cout.setf(ios::fixed, ios::floatfield); // 设置使用十进制计数法，不使用科学计数法
	cout << "调用核函数时间："  << double(duration.count()) * microseconds::period::num / microseconds::period::den << "秒" << endl;
	// ===========================================================
}

/// <summary>
/// 主函数
/// </summary>
/// <param name="">void</param>
/// <returns>int</returns>
int main(void) {

	for (int n = 0; n < 10; n += 1)
	{
		for (int m = 0; m < 10; m += 1)
		{
			runKernel(n, m);
			cout << endl;
		}
	}

	return 0;
}