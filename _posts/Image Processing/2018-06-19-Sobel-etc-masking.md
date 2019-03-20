---
layout: post
title: window masking 중 sobel, laplacian, gausian 적용해보자
date: 2018-06-18 08:30:00 +0900
description: mfc imageprocessing window masking 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
categories: [Image Processing]
---

이전 작업에 이어서, 다양한 윈도우 마스킹을 적용해보자.

먼저 sobel masking.

> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void SobelMasking(unsigned char* image_color,
		const int width, const int height);
}
{% endhighlight %}

> ImageProc.cpp
mask만 sobel로 바꾸고 이전 함수를 이용하면 된다.

{% highlight cpp %}
void ImageProc::SobelMasking(unsigned char* image_color,
	const int width, const int height)
{
	float mask_sobel[5][5] = {
		{ -1.f, -1.f, 0.f, 1.f, 1.f },
		{ -1.f, -1.f, 0.f, 1.f, 1.f },
		{ -2.f, -2.f, 0.f, 2.f, 2.f },
		{ -1.f, -1.f, 0.f, 1.f, 1.f },
		{ -1.f, -1.f, 0.f, 1.f, 1.f },
	};

	unsigned char* image_R = new unsigned char[width*height];
	unsigned char* image_G = new unsigned char[width*height];
	unsigned char* image_B = new unsigned char[width*height];

	SplitChannels_ColorToRGB(image_R, image_G, image_B, image_color, width, height);

	MaskingImage5x5(image_R, width, height, mask_sobel);
	MaskingImage5x5(image_G, width, height, mask_sobel);
	MaskingImage5x5(image_B, width, height, mask_sobel);

	MergeChannels_RGBToColor(image_R, image_G, image_B, image_color, width, height);

	delete[] image_R;
	delete[] image_G;
	delete[] image_B;
}
{% endhighlight %}

이벤트 처리기를 단다.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc: public CDocument
{
public:
    afx_msg void OnWindowmaskingSobel();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnWindowmaskingSobel()
{
    ImageProc::SobelMasking(m_Images[cur_index].image_color,
        m_Images[cur_index].width, m_Images[cur_index].height);
    
    // 작업한 칼라이미지를 흑백이미지에도 복사
    ImageProc::MergeChannels(m_Images[cur_index].image_color,
        m_Images[cur_index].image_gray, m_Images[cur_index].width, m_Images[cur_index].height);
    
    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

결과는 다음과 같다.

![sobel5x5-hijy]({{"/assets/img/imageProcessing/sobel5x5-hijy.jpg"}})

마스크만 숫자를 변경해서 laplacian, gausian masking을 할 수 있다.

laplacian masking
{% highlight cpp%}
float mask_laplacian[5][5] = {
		{ -4.f, -1.f, 0.f, -1.f, -4.f },
		{ -1.f,  2.f, 3.f,  2.f, -1.f },
		{  0.f,  3.f, 4.f,  3.f,  0.f },
		{ -1.f,  2.f, 3.f,  2.f, -1.f },
		{ -4.f, -1.f, 0.f, -1.f, -4.f },
};
{% endhighlight %}
![laplacian5x5-hijy]({{"/assets/img/imageProcessing/laplacian5x5-hijy.jpg"}})

gausian masking
{% highlight cpp%}
float mask_gausian[5][5] = {
		{ 1.f / 273.f,	4.f / 273.f,  7.f / 273.f, 4.f  / 273.f, 1.f / 273.f },
		{ 4.f / 273.f, 16.f / 273.f, 26.f / 273.f, 16.f / 273.f, 4.f / 273.f },
		{ 7.f / 273.f, 26.f / 273.f, 41.f / 273.f, 26.f / 273.f, 7.f / 273.f },
		{ 4.f / 273.f, 16.f / 273.f, 26.f / 273.f, 16.f / 273.f, 4.f / 273.f },
		{ 1.f / 273.f,	4.f / 273.f,  7.f / 273.f, 4.f  / 273.f, 1.f / 273.f },
	};
{% endhighlight %}
![gausian5x5-hijy]({{"/assets/img/imageProcessing/gausian5x5-hijy.jpg"}})
