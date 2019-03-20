---
layout: post
title: mfc-imageProcessing masking을 해보자
date: 2018-06-01 09:00:00 +0900
description: mfc imageprocessing masking 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
categories: [Image Processing]
---

이전 [바이너리화](../binarization-etc), [침식,팽창](../imageErosion-Dilation)까지 픽셀단위, 영역단위 이미지 조작을 했다.

이번 포스트에서는 마스킹하는 작업을 해보자.

이미지와 같은 크기의 마스크를 만들고 칼라 이미지에 마스킹을 적용 할 것이다.

먼저 왼쪽에서 부터 오른쪽으로 갈 수록 어두워지는 마스크를 만들고 적용해보자.
마스크는 0 부터 1사이의 실수를 값으로 갖는다.

> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void UserMasking1(unsigned char* image_color,
        const int width, const int height);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::UserMasking1(unsigned char* image_color,
    const int width, const int height)
{   
    // mask 를 만든다. size는 image_color 와 같다.
    float* mask = new float[width*height];

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            mask[width*j + i] = static_cast<float>(width-i) / static_cast<float>(width);
        }
    }

    // image_color 에 masking 한다.
    for(int i=0; i<width*height; i++)
    {
        image_color[i*4 + 0] *= mask[i]; 
        image_color[i*4 + 1] *= mask[i]; 
        image_color[i*4 + 2] *= mask[i]; 
    }

    delete[] mask;
}
{% endhighlight %}

이벤트 처리기를 만들자.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc:public CDocument
{
    ...
public:
    afx_msg void OnMaskingUsermasking1();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnMaskingUsermasking1()
{
    ImageProc::UserMasking1(m_Images[cur_index].image_color, 
		m_Images[cur_index].width, m_Images[cur_index].height);
	ImageProc::MergeChannels(m_Images[cur_index].image_color,
		m_Images[cur_index].image_gray,m_Images[cur_index].width, m_Images[cur_index].height);

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 아래와 같이 보인다
![userMasking1-jy]({{"/assets/img/imageProcessing/userMasking1-jy.jpg"}})

이번엔 원 모양의 마스크를 만들어보자. 가운데가 밝고 가장자리로 갈수록 어두워진다.
> ImageProc.h
{% highlight cpp %}
...
// 절대값과 제곱루트 함수를 사용하기 위해 math.h 를 추가한다.
#include <math.h>
class ImageProc
{
    ...
public:
    static void UserMaskingCircle(unsigned char* image_color,
        const int width, const int height);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::UserMaskingCircle(unsigned char* image_color,
    const int width, const int height)
{
    float* mask = new float[width*height];

    // 원의 중심
    int centerX = width / 2;
    int centerY = height / 2;
    // 중심으로부터 거리
    float distanceFromCenterX = 0.f;
    float distanceFromCenterY = 0.f;
    float distanceFromCenter = 0.f;

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            distanceFromCenterX = fabsf(static_cast<float>(i) - centerX);
            distanceFromCenterY = fabsf(static_cast<float>(i) - centerY);
            distanceFromCenter = sqrt((distanceFromCenterX*distanceFromCenterX) + (distanceFromCenterY*distanceFromCenterY));

            // 거리에 비례해 mask에 값을 넣음.
            // 중심 0 부터 바깥쪽 1임.
            mask[width*j +i] = ((width / 2) - distanceFromCenter) / (width / 2);
            if(mask[width*j +i] < 0)
                mask[width*j + i] = 0;
        }
    }

    // masking 적용
    for(int i=0; i<width*height; i++)
    {
        image_color[i*4 + 0] *= mask[i];
        image_color[i*4 + 1] *= mask[i];
        image_color[i*4 + 2] *= mask[i];
    }

    delete[] mask;
}
{% endhighlight %}

이벤트 처리기를 추가한다.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc:public CDocument
{
    ...
public:
    afx_msg void OnUsermasking2Circle();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnUsermasking2Circle()
{
    ImageProc::UserMaskingCircle(m_Images[cur_index].image_color,
        m_Images[cur_index].width, m_Images[cur_index].height);
    ImageProc::MergeChannels(m_Images[cur_index].image_gray,
        m_Images[cur_index].width, m_Images[cur_index].height);
    
    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

    pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 아래와 같이 보인다
![userMaskingCircle-jy]({{"/assets/img/imageProcessing/userMaskingCircle-jy.jpg"}})

이번엔 위가 밝고 아래는 어두운 마스크를 만들어보자.

> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void UserMaskingUpDown(unsigned char* image_color,
        const int width, const int height);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::UserMaskingUpDown(unsigned char* image_color,
    const int width, const int height)
{   
    // mask 를 만든다. size는 image_color 와 같다.
    float* mask = new float[width*height];

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            mask[width*j + i] = static_cast<float>(height-j) / static_cast<float>(height);
        }
    }

    // image_color 에 masking 한다.
    for(int i=0; i<width*height; i++)
    {
        image_color[i*4 + 0] *= mask[i]; 
        image_color[i*4 + 1] *= mask[i]; 
        image_color[i*4 + 2] *= mask[i]; 
    }

    delete[] mask;
}
{% endhighlight %}

이벤트 처리기를 만들자.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc:public CDocument
{
    ...
public:
    afx_msg void OnUsermasking2Updown();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnUsermasking2Updown()
{
    ImageProc::UserMaskingUpDown(m_Images[cur_index].image_color, 
		m_Images[cur_index].width, m_Images[cur_index].height);
	ImageProc::MergeChannels(m_Images[cur_index].image_color,
		m_Images[cur_index].image_gray,m_Images[cur_index].width, m_Images[cur_index].height);

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 아래와 같이 보인다
![userMaskingUpDown-jy]({{"/assets/img/imageProcessing/userMaskingUpDown-jy.jpg"}})


이번엔 화면을 세로로 16등분해서 행이 0 과 1을 번갈아 갖고 있는 마스크를 만들어보자.
> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void UserMaskingDivide(unsigned char* image_color,
        const int width, const int height);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::UserMaskingDivide(unsigned char* image_color,
    const int width, const int height)
{
    unsigned char* mask = new unsigned char[width*height];
    int changePoint = height / 16;
    int bBlack = -1;

    for(int j=0; j<height; j++)
    {
        if(j % changePoint == 0)
            bBlack *= -1;

        for(int i=0; i<width; i++)
        {
            if(bBlack == 1)
                mask[width*j + i] = 0;
            else
                mask[width*j + i] = 1;
        }
    }

    for(int i=0; i<width*height; i++)
    {
        image_color[i*4 + 0] *= mask[i];
        image_color[i*4 + 1] *= mask[i];
        image_color[i*4 + 2] *= mask[i];
    }

    delete[] mask;
    
}
{% endhighlight %}

이벤트 처리기를 만들자.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc:public CDocument
{
    ...
public:
    afx_msg void OnUsermasking2Divide();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnUsermasking2Divide()
{
    ImageProc::UserMaskingDivide(m_Images[cur_index].image_color, 
		m_Images[cur_index].width, m_Images[cur_index].height);
	ImageProc::MergeChannels(m_Images[cur_index].image_color,
		m_Images[cur_index].image_gray,m_Images[cur_index].width, m_Images[cur_index].height);

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 아래와 같이 보인다
![userMaskingDivide-jy]({{"/assets/img/imageProcessing/userMaskingDivide-jy.jpg"}})

위 마스킹을 RGB 별로 각각 적용해보자.

> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void UserMaskingCUD(unsigned char* image_color,
        const int width, const int height);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::UserMaskingCUD(unsigned char* image_color,
    const int width, const int height)
{
	// mask_circle
	float* mask_circle = new float[width*height];
	int centerX = width / 2;
	int centerY = height / 2;
	float distanceFromCenterX = 0.f;
	float distanceFromCenterY = 0.f;
	float distanceFromCenter = 0.f;

	// 중심으로 부터 거리를 구하고
	// 거리에 비례해 mask 값을 채운다.
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			distanceFromCenterX = fabsf((static_cast<float>(i) - centerX));
			distanceFromCenterY = fabsf((static_cast<float>(j) - centerY));
			distanceFromCenter = sqrt((distanceFromCenterX*distanceFromCenterX)
				+ (distanceFromCenterY*distanceFromCenterY));

			mask_circle[width*j + i] = ((width / 2) - distanceFromCenter) / (width / 2);
			if (mask_circle[width*j + i] < 0)
				mask_circle[width*j + i] = 0;
		}
	}


	// mask_updown
	float* mask_updown = new float[width*height];

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			mask_updown[width*j + i] = static_cast<float>(height - j) /
				static_cast<float>(height);
		}
	}

	// mask_divide
	unsigned char* mask_divide = new unsigned char[width*height];
	int changePoint = height / 16;
	int bZero = -1;

	for (int j = 0; j<height; j++)
	{
		if (j % changePoint == 0)
			bZero *= (-1);

		for (int i = 0; i<width; i++)
		{
			if (bZero == 1)
				mask_divide[width*j + i] = 0;
			else
				mask_divide[width*j + i] = 1;
		}
	}

	// masking
	for (int i = 0; i < width*height; i++)
	{
		image_color[i * 4 + 0] *= mask_updown[i];
		image_color[i * 4 + 1] *= mask_divide[i];
		image_color[i * 4 + 2] *= mask_circle[i];
	}

	delete[] mask_circle;
	delete[] mask_updown;
	delete[] mask_divide;
}
{% endhighlight %}

이벤트 처리기를 만들자.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc:public CDocument
{
    ...
public:
    afx_msg void OnUsermasking2Cud();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnUsermasking2Cud()
{
    ImageProc::UserMaskingCUD(m_Images[cur_index].image_color, 
		m_Images[cur_index].width, m_Images[cur_index].height);
	ImageProc::MergeChannels(m_Images[cur_index].image_color,
		m_Images[cur_index].image_gray,m_Images[cur_index].width, m_Images[cur_index].height);

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 아래와 같이 보인다
![userMaskingCUD-jy]({{"/assets/img/imageProcessing/userMaskingCUD-jy.jpg"}})