---
layout: post
title: mfc-imageProcessing 화면에 흑백이미지를 함께 출력해보자
date: 2018-05-25 09:27:00 +0900
description: mfc imageprocessing 화면에 흑백이미지를 함께 출력해보자 # Add post description (optional)
img:  # Add image post (optional)
---

기존 ImageProcessing [작업](../mfcImageProcessing) 에서는 화면에 그림 하나를 띄웠다.
이번에는 화면에 원본그림 하나와 흑백이미지 그림 하나를 함께 띄우는 방법을 알아보자.

순서는 이렇다. 먼저 이미지를 읽는다. -> 이미지를 흑백으로 만든다. -> 원본과 흑백을 함께 띄운다.

현재 화면에 그림을 그리는 방법을 다르게 해야한다. 원본 이미지의 2배 만큼 길이를 잡고 왼쪽에는 원본을 오른쪽에는 흑백 그림을 띄운다.
이를 위해 SetDraw 함수를 변경해보자.

기존의 함수는 아래와 같다.
{% highlight cpp %}
void CImageProcessingView::SetDrawImage(unsigned char* image,
	const int width, const int height, const int byte)
{
	if (byte<=0) return;

	m_ImgWidth = width;
	m_ImgHeight = height;

	if (m_Image)
		delete[] m_Image;

	m_Image = new unsigned char[width*height * 4];

	if (byte > 1)
	{
		for (int i = 0; i < width*height; i++)
		{
			m_Image[i * 4 + 0] = image[i * byte + 0];
			m_Image[i * 4 + 1] = image[i * byte + 1];
			m_Image[i * 4 + 2] = image[i * byte + 2];
		}
	}
	else
	{
		for (int i = 0; i < width*height; i++)
		{
			m_Image[i * 4 + 0] = image[i];
			m_Image[i * 4 + 1] = image[i];
			m_Image[i * 4 + 2] = image[i];
		}
	}
}
{% endhighlight %}

이제 위 함수에서 gray 이미지도 받아 화면에 그릴것이다. 함수를 overload 해보자.
{% highlight cpp %}
void CImageProcessingView::SetDrawImage(unsigned char* image, unsigned char* image_gray,
	const int width, const int height)
{
    // byte 는 따로 받지 않는다.
    m_ImgWidth = width * 2;
    m_ImgHeight = height;

    if(m_Image)
        delete [] m_Image;
	
	// 화면을 두 배로 써야 하기 때문에 *2를 해준다.
    m_Image = new unsigned char[width*2*height*4];

    for(int i=0; i<width*height; i++)
    {
        m_Image[i*4 + 0] = image[i*4 + 0];
        m_Image[i*4 + 1] = image[i*4 + 1];
        m_Image[i*4 + 2] = image[i*4 + 2];
    }

	// color image 다음에 gray image 가 나오므로 + width 를 한다.
    for(int i=0; i<width*height; i++)
    {
        m_Image[i*4 + width + 0] = image_gray[i*4];
		m_Image[i*4 + width + 1] = image_gray[i*4];
		m_Image[i*4 + width + 2] = image_gray[i*4];
    }
}
{% endhighlight %}

이제 color image를 gray image 로 바꾸는 함수를 정의해보자. ImageProc.cpp 에 작성한다.
{% highlight cpp %}
// gray image의 픽셀에 color image rgb 값을 평균하여 저장하면 된다.

void ImageProc::MergeChannels(unsigned char* in_color,
	unsigned char* out_gray, const int width, const int height)
{
	for(int i=0; i<width*height; i++)
	{
		out_gray[i] = (in_color[i*4 + 0] + in_color[i*4 + 1] + in_color[i*4 +2]) / 3;
	}
}
{% endhighlight %}

이제 Doc 에 이벤트 처리기를 만들면 되는데, 파일을 오픈할때 gray 처리를 해서 color와 함께 출력되게 만들자.
{% highlight cpp %}
void CImageProcessingDoc::OnFileOpen()
{
	
}
{% endhighlight %}