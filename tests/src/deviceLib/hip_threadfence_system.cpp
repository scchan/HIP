/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <thread>
#include <cassert>
#include <cstdio>
#include "hip/hip_runtime.h"
#include <hip/device_functions.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void round_robin(const int id, const int num_dev, const int N, int* data, int* flag) {
  for (int i = 0; i < N; i++) {
    while(*flag%num_dev != id) ;

    (*data) += 1;
    __threadfence_system();
    (*flag) += 1;
    __threadfence_system();
  }
}

int main() {

  int* data;
  int* flag;

  HIP_ASSERT(hipMalloc(&data, sizeof(int)));
  int init_value = 1000;
  HIP_ASSERT(hipMemcpy(data, &init_value, sizeof(int), hipMemcpyHostToDevice)); 

  HIP_ASSERT(hipMalloc(&flag, sizeof(int)));
  init_value = 0;
  HIP_ASSERT(hipMemcpy(flag, &init_value, sizeof(int), hipMemcpyHostToDevice)); 

  constexpr int num_dev = 2;
  std::vector<std::thread> threads;
  dim3 dim_block(1,1,1);
  dim3 dim_grid(1,1,1);
  for (int i = 0; i < num_dev; i++) {
    threads.push_back(std::thread([=]() {
      HIP_ASSERT(hipSetDevice(i));
      hipLaunchKernelGGL(round_robin, dim_grid, dim_block, 0, 0x0
                          , i, num_dev, 2, data, flag);
    }));
  }

  for (auto& t : threads) {
    t.join();
  }

  printf("data: %d\n", *data);

  return 0;
}
