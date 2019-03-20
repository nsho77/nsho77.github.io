---
layout: post
title: CUDA 이용하여 행렬 곱셈을 해보자
date: 2018-07-03 09:00:00 +0900
description: Cuda Progamming # Add post description (optional)
img:  # Add image post (optional)
tags: [development, Cuda]
categories: [Image Processing]
---

CUDA 를 이용해 행렬의 곱셈을 해보자.

> kernel.cu
{% highlight cpp %}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Stdio.h>
// 행렬 곱셈 커널 함수를 콜할 호스트 함수
cudaError_t multiWithCuda(float* c, float* a, float* b, unsigned int size);

__global__ void multiKernel(float* c, float* a, float* b, unsigned int size)
{
    int i = threadIdx.x;
    // block 갯수를 지정해 idx를 활용할 것이다.
    // (1차원 배열을 사용할 것이지만)
    // 행렬은 2차원이기 때문이다.
    int j = blockIdx.x;
    
    // 행렬의 곱셈을 구현한다.
    for(int x=0; x<size; x++)
        c[size*j + i] += a[size*j + x] * b[size*x + i];
}

int main()
{
    const int arraySize = 5;
    // 행렬 a,b,c 를 만든다.
    float a[arraySize*arraySize] = {0};
    float b[arraySize*arraySize] = {0};
    float c[arraySize*arraySize] = {0};

    // 알맞은 값으로 초기화 한다.
    for(int i=0; i<arraySize*arraySize; i++)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    // 작업할 함수를 콜한다.
    cudaError_t cudaStatus = multiWithCuda(c, a, b, arraySize);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "multiWithCuda failed!");
        return -1;
    }

    // 결과를 출력한다.
    for(int i=0; i<arraySize*arraySize; i++)
    {
        if(i % arraySize == 0) printf("\n");
        printf("%8.1f ",c[i]);
    }
    printf("\n");

    // 모든 작업이 완료되었으므로
    // device 를 reset 한다.
    cudaStatus = cudaDeviceReset();
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaDeviceReset, failed!");
        return 1;
    }

    return 0;
}

// 커널함수 호출하는 헬퍼 함수 multiWithCuda를 정의하자
cudaError_t multiWithCuda(float* c, float* a, float* b, unsigned int size)
{
    // gpu에 할당한 메모리 주소값을 저장할 변수를 선언한다.
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // GPU에 메모리를 할당한다.
    // 행렬 크기만큼 할당한다.
    cudaStatus = cudaMalloc((void**)&dev_c, size*size*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_a, size*size*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, size*size*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc failed!");
        goto Error;
    }

    // 호스트 메모리에 있는 값을 디바이스 메모리에 복사한다.
    cudaStatus = cudaMemcpy(dev_a, a, size*size*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
     cudaStatus = cudaMemcpy(dev_b, b, size*size*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // 커널 함수를 실행한다.
    multiKernel<<<size, size>>>(dev_c, dev_a, dev_b, size);

    // 커널 함수 실행후 에러가 있는지 확인
    cudataStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "multiKernel launch failed : %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 커널이 모두 종료되었는지 확인
    cudaStatus  = cudaDeviceSynchronize();
    if(cudaStatus!= cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // 결과를 호스트 메모리에 복사
    cudaStatus = cudaMemcpy(c, dev_c, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
// gpu에 할당한 메모리를 반환
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
{% endhighlight %}

결과는 다음과 같다  
```
     150      160      170      180      190  
     400      435      470      505      540  
     650      710      770      830      890  
     900      985     1070     1155     1240  
    1150     1260     1370     1480     1590
```