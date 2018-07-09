---
layout: post
title: Binary Dilation, Erosion 을 GPU로 처리해보자
date: 2018-07-09 10:00:00 +0900
description: image processing 을 gpu를 이용해 처리해보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, Cuda]
---


이번 포스팅에서 기존 영상처리에 사용되었던 기술인 Adaptive Binarization 을 
CUDA 를 이용해 구현하는 방법을 알아볼 것이다.    

그 전 작업에서는 GPU 메모리 할당 작업을 .cu 안에서 진행했다. 그래서 ImageProcessingDoc.cpp 이벤트 함수에
메모리 할당 작업은 없고 메모리 해제 작업만 있었다. 이를 Doc 에서 할당하고 Doc 에서 해제 하게끔 바꿔보자.

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnImageprocessinggpuAdaptivebinarization()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	printf("OnImageprocessing GPU Adaptivebinarization\n");
    // 메모리를 할당하는 코드를 추가한다.
	obj_ImageProc->AllocateGPUMemory(m_Images[cur_index].width, m_Images[cur_index].height);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&BeginTime);

	obj_ImageProc->GPU_AdaptiveBinarization(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 125);
...
}
{% endhighlight %}

.cu 파일에 불필요하게 메모리를 할당하고 0으로 세팅하는 코드는 삭제한다.
> CUDAImageProc.cu
{% highlight cpp %}
extern "C"
int ImageProc_AdaptiveBinarization(unsigned char* image_gray,
	int width, int height, int ksize)
{
    // 이 부분을 삭제한다.
	// GPU에 메모리 할당하고 0으로 세팅한다.
	/*if (ImageProc_AllocGPUMemory(width, height) < 0)
		return -1;*/

	// GPU 메모리에 호스트 데이터 복사한다.
	cudaError cudaStatus;
	cudaStatus = cudaMemcpy(g_tempBuffer[0],image_gray,
		sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -2;
        ...
}
{% endhighlight %}

이제 Doc 에서 GPU 메모리를 할당하고, 해제하게 되었다.    
그러면 Binary Dilation, Erosion 을 구현해보자.

이벤트 처리기 부터 만들어보자
> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnImageprocessinggpuBinarydilation()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	obj_ImageProc->AllocateGPUMemory(m_Images[cur_index].width, m_Images[cur_index].height);

	obj_ImageProc->GPU_BinaryDilation(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 5);

	if (obj_ImageProc->DeAllocateGPUMemory())
		printf("DeAllocation success!!\n");

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 1);

	pView->OnInitialUpdate();
}


void CImageProcessingDoc::OnImageprocessinggpuBinaryerosion()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	obj_ImageProc->AllocateGPUMemory(m_Images[cur_index].width, m_Images[cur_index].height);

	obj_ImageProc->GPU_BinaryErosion(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 5);

	if (obj_ImageProc->DeAllocateGPUMemory())
		printf("DeAllocation success!!\n");

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 1);

	pView->OnInitialUpdate();
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
bool ImageProc::GPU_BinaryDilation(unsigned char* image_binary,
	int width, int height, int ksize)
{
	printf("GPU_BinaryDilation\n");
	int res = ImageProc_BinaryDilation(image_binary, width, height, ksize);
	if (res < 0)
	{
		printf("GPU_BinaryDilation failed error code is %d \n,", res);
		return false;
	}
	else
		return true;
}

bool ImageProc::GPU_BinaryErosion(unsigned char* image_binary,
	int width, int height, int ksize)
{
	printf("GPU_BinaryErosion\n");
	int res = ImageProc_BinaryErosion(image_binary, width, height, ksize);
	if (res < 0)
	{
		printf("GPU_BinaryErosion failed error code is %d \n,", res);
		return false;
	}
	else
		return true;
}
{% endhighlight %}

ImageProc 에서 사용할 .cu 소스의 함수를 선언하자.
> ImageProc.h
{% highlight cpp %}
extern "C"
{
    ...
    int ImageProc_BinaryDilation(unsigned char* image_binary,
		int width, int height, int ksize);
	int ImageProc_BinaryErosion(unsigned char* image_binary,
		int width, int height, int ksize);
}
{% endhighlight %}

.cu 에 커널 함수와 커널함수 호출을 도와줄 함수를 정의하자.
> CUDAImageProc.cu

{% highlight cpp %}
__global__ void Kernel_BinaryDilation(unsigned char* image_binary,
	unsigned char* output_image, int width, int height, int ksize)
{
	if (ksize == 1 || ksize % 2 == 0) return;
	int neighbor = ksize / 2;
	// blockIdx, blockDim, threadIdx 로 좌표를 찾는다.
	// blockDim 은 block 사이즈 이므로 아래와 같이 좌표를 구할 수 있다.
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	// 현재 픽셀이 255 이면 별도의 작업을 하지 않는다.
	if (image_binary[width*j + i] == 255) return;

	// 주변 픽셀이 255 이면 현재 픽셀을 255로 바꾼다.
	for (int x = -neighbor; x <= neighbor; x++)
	{
		for (int y = -neighbor; y <= neighbor; y++)
		{
			if (i + x < 0 || i + x >= width || j + y < 0 || j + y >= height)
				continue;
			if (image_binary[width*(y + j) + i + x] == 255)
			{
				output_image[width*j + i] = 255;
				return;
			}
			
		}
	}

}

__global__ void Kernel_BinaryErosion(unsigned char* image_binary,
	unsigned char* output_image, int width, int height, int ksize)
{
	if (ksize == 1 || ksize % 2 == 0) return;
	int neighbor = ksize / 2;
	// blockIdx, blockDim, threadIdx 로 좌표를 찾는다.
	// blockDim 은 block 사이즈 이므로 아래와 같이 좌표를 구할 수 있다.
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	// 현재 픽셀이 0 이면 별도의 작업을 하지 않는다.
	if (image_binary[width*j + i] == 0) return;

	// 주변 픽셀이 0 이면 현재 픽셀을 0로 바꾼다.
	for (int x = -neighbor; x <= neighbor; x++)
	{
		for (int y = -neighbor; y <= neighbor; y++)
		{
			if (i + x < 0 || i + x >= width || j + y < 0 || j + y >= height)
				continue;
			if (image_binary[width*(y + j) + i + x] == 0)
			{
				output_image[width*j + i] = 0;
				return;
			}

		}
	}

}

extern "C"
int ImageProc_BinaryDilation(unsigned char* image_binary,
	int width, int height, int ksize)
{
	// GPU 메모리에 호스트 데이터 복사한다.
	cudaError cudaStatus;
	cudaStatus = cudaMemcpy(g_tempBuffer[0], image_binary,
		sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -2;

	cudaStatus = cudaMemcpy(g_tempBuffer[1], image_binary,
		sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -2;

	// 커널 함수를 실행한다.
	// GridDim.x, GridDim.y, BlockDim.x, BlockDim.y 를 정의한다.
	dim3 Db = dim3(8, 8);
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
	Kernel_BinaryDilation << < Dg, Db >> > (g_tempBuffer[0],
		g_tempBuffer[1], width, height, ksize);

	// 커널 함수 실행이 제대로 되었는지 확인한다.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return -3;


	// 커널 함수 모두 종료를 확인한다.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaStatus : %d\n", cudaStatus);
		return -4;
	}


	// 결과를 호스트 메모리로 복사한다.
	cudaStatus = cudaMemcpy(image_binary, g_tempBuffer[1],
		sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		return -5;

	return 1;

}


extern "C"
int ImageProc_BinaryErosion(unsigned char* image_binary,
	int width, int height, int ksize)
{
	// GPU 메모리에 호스트 데이터 복사한다.
	cudaError cudaStatus;
	cudaStatus = cudaMemcpy(g_tempBuffer[0], image_binary,
		sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -2;

	cudaStatus = cudaMemcpy(g_tempBuffer[1], image_binary,
		sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -2;

	// 커널 함수를 실행한다.
	// GridDim.x, GridDim.y, BlockDim.x, BlockDim.y 를 정의한다.
	dim3 Db = dim3(8, 8);
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
	Kernel_BinaryErosion << < Dg, Db >> > (g_tempBuffer[0],
		g_tempBuffer[1], width, height, ksize);

	// 커널 함수 실행이 제대로 되었는지 확인한다.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return -3;


	// 커널 함수 모두 종료를 확인한다.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaStatus : %d\n", cudaStatus);
		return -4;
	}


	// 결과를 호스트 메모리로 복사한다.
	cudaStatus = cudaMemcpy(image_binary, g_tempBuffer[1],
		sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		return -5;

	return 1;

}
{% endhighlight %}

결과는 기존의 Dilation, Erosion 과 같아야 한다.