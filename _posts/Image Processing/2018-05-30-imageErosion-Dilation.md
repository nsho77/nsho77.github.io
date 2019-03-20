---
layout: post
title: mfc-imageProcessing 이미지 팽창, 침식기능을 추가하자
date: 2018-05-30 09:00:00 +0900
description: mfc imageprocessing 다양한 이미지 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
categories: [Image Processing]
---

# 픽셀 단위 작업

* 밝기 조절
* 콘트라스트 조절
* 흑백이미지를 바이너리화

# 영역 단위 작업

* 이진 이미지 침식
* 이진 이미지 팽창

[이전](../binarization-etc)까지 픽셀단위 이미지를 조작하는 작업을 했다.

이번 포스트에서는 영역단위 이미지를 조작하는 작업을 해보자.


이진 이미지 침식이란 바이너리 이미지를 주변 x 픽셀만큼 보고 검은색이 있다면 검게 만드는 작업이다.
이진 이미지 팽창이란 바이너리 이미지를 주변 x 픽셀만큼 보고 하얀색이 있다면 하얗게 만드는 작업이다.

주로 이미지의 노이즈를 줄일 때 사용한다.


#### 이진 이미지 침식

> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public :
    static void BinaryErosion(unsigned char* image_gray,
        const int width, const int height, unsigned char ksize);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
// 현재 픽셀이 검은색이면 건너뛴다.
// 주변 x 픽셀만큼 보고 검은색이 있다면 검게 만든다.
void ImageProc::BinaryErosion(unsigned char* image_gray,
    const int width, const int height, unsigned char ksize)
{
    // ksize 는 홀수로 받아서 양옆 neighbor 만큼 본다.
    unsigned char neighbor  = ksize /2 ;

    // 침식이미지를 저장할 버퍼를 생성한다.
    unsigned char* temp = new unsigned char[width*height];
    memcpy(temp, image_gray, sizeof(unsigned char)*width*height);

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            //현재 픽셀이 검은색이면 건너뛴다.
            if(image_gray[width*j + i] == 0)
            {
                temp[width*j + i] =0;
                continue;
            }

            // 주변 픽셀이 검은색인지 확인
            bool bBlack = false;

            for( int l = j-neighbor; l< j+1+neighbor; l++)
            {
                for (int k = i-neighbor; k<i+1+neighbor; k++)
                {
                    if(image_gray[width*l + k] == 0)
                    {
                        temp[width*j + i]= 0;
                        bBlack = true;
                        break;
                    }
                }
                if(bBlack)
                    break;
            }
        }
    }

    memcpy(image_gray, temp, sizeof(unsigned char)*width*height);
    delete[] temp;
}
{% endhighlight %}

이벤트를 추가한다.
> ImageProcessingDoc.h
{% highlight cpp %}
class ImageProcessingDoc : public CDocument
{
    ...

public:
    afx_msg void OnAreaprocessingBinaryerosion();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnAreaprocessingBinaryerosion()
{
    // 먼저 binarization 한다.
    ImageProc::Binarization(m_Images[cur_index].image_gray, 
        m_Images[cur_index].width, m_Images[cur_index].height, 127);

    // 주변 1개의 픽셀을 바라본다.
    ImageProc::BinaryErosion(m_Images[cur_index].image_gray,
        m_Images[cur_index].width, m_Images[cur_index].height, 3);

    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

    pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 다음 그림과 같이 보인다.
![binaryerosion-jy]({{"/assets/img/imageProcessing/binaryerosion-jy.jpg"}})

#### 이진 이미지 팽창
> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public :
    static void BinaryDilation(unsigned char* image_gray,
        const int width, const int height, unsigned char ksize);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
// 현재 픽셀이 하얀색이면 건너뛴다.
// 주변 x 픽셀만큼 보고 하얀색이 있다면 하얗게 만든다.
void ImageProc::BinaryDilation(unsigned char* image_gray,
    const int width, const int height, unsigned char ksize)
{
    // ksize 는 홀수로 받아서 양옆 neighbor 만큼 본다.
    unsigned char neighbor  = ksize /2 ;

    // 팽창이미지를 저장할 버퍼를 생성한다.
    unsigned char* temp = new unsigned char[width*height];
    memcpy(temp, image_gray, sizeof(unsigned char)*width*height);

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            //현재 픽셀이 하얀색이면 건너뛴다.
            if(image_gray[width*j + i] == 0)
            {
                temp[width*j + i] =0;
                continue;
            }

            // 주변 픽셀이 하얀색인지 확인
            bool bWhite = false;

            for( int l = j-neighbor; l< j+1+neighbor; l++)
            {
                for (int k = i-neighbor; k<i+1+neighbor; k++)
                {
                    if(image_gray[width*l + k] == 0)
                    {
                        temp[width*j + i]= 0;
                        bWhite = true;
                        break;
                    }
                }
                if(bWhite)
                    break;
            }
        }
    }

    memcpy(image_gray, temp, sizeof(unsigned char)*width*height);
    delete[] temp;
}
{% endhighlight %}

이벤트를 추가한다.
> ImageProcessingDoc.h
{% highlight cpp %}
class ImageProcessingDoc : public CDocument
{
    ...

public:
    afx_msg void OnAreaprocessingBinarydilation();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnAreaprocessingBinarydilation()
{
    // 먼저 binarization 한다.
    ImageProc::Binarization(m_Images[cur_index].image_gray, 
        m_Images[cur_index].width, m_Images[cur_index].height, 127);

    // 주변 1개의 픽셀을 바라본다.
    ImageProc::BinaryDilation(m_Images[cur_index].image_gray,
        m_Images[cur_index].width, m_Images[cur_index].height, 3);

    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

    pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 아래와 같이 보인다
![binarydilation-jy]({{"/assets/img/imageProcessing/binarydilation-jy.jpg"}})