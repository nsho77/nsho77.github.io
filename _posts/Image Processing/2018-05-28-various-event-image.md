---
layout: post
title: mfc-imageProcessing 다양한 이미지 작업을 알아보자
date: 2018-05-28 16:00:00 +0900
description: mfc imageprocessing 다양한 이미지 작업을 알아보자 # Add post description (optional)
img:   # Add image post (optional)
tags: [development, imageProcessing]
categories: [Image Processing]
---

[이전](../ImageProcessing-grayImage)까지 칼라이미지와 흑백이미지를 함께 출력하는 작업을 했다.

다양한 이미지 작업을 해보자. 작업 목록은 아래와 같다.

# 픽셀 단위 작업

* 밝기 조절
* 콘트라스트 조절
* 흑백이미지를 바이너리화

# 영역 단위 작업

* 이진 이미지 침식
* 이진 이미지 팽창

이 포스트에서는 밝기 조절과 콘트라스트 조절 작업에 대해 알아볼 것이다.

#### 밝기조절

ImageProc.h, ImageProc.cpp 에 다음과 같은 밝기 조절 함수를 추가한다.
> ImageProc.h
{% highlight cpp %}
class ImageProc{
    ...

public:
    static void AdjustBright(unsigned char* image,
        const int width, const int height, const int bright );
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::AdjustBright(unsigned char* image,
        const int width, const int height, const int bright )
{
    // 255 보다 크면 255 0보다 작으면 0을 저장한다.
    for(int i=0; i<width*height; i++)
    {
        image[i*4 + 0] = static_cast<char>(min(255,(max(0,(image[i*4 + 0] + bright)))));
        image[i*4 + 1] = static_cast<char>(min(255,(max(0,(image[i*4 + 1] + bright)))));
        image[i*4 + 2] = static_cast<char>(min(255,(max(0,(image[i*4 + 2] + bright)))));
    }
}
{% endhighlight%}

이제 이벤트를 처리하는 함수를 정의한다.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc::public CDocument
{
    ...
public :
    afx_msg void OnPixelProcessingAdjustbright();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
...
void OnPixelProcessingAdjustbright()
{
    ImageProc::AdjustBright(m_Images[cur_index].image_color, 
        m_Images[cur_index].width, m_Images[cur_index].height, 34);

    // 흑백 이미지 밝기도 조절 한다.
    ImageProc::MergeChannels(m_Images[cur_index].image_color, 
       m_Images[cur_index].image_gray, m_Images[cur_index].width, m_Images[cur_index].height);
    
    // View 포인터를 얻어오고 이미지를 세팅한다.
    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();
    pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

    // 화면을 업데이트 한다.
    pView->OnInitialUpdate();
}
...
{% endhighlight %}


![panda-color-gray]({{"/assets/img/imageProcessing/color-gray-panda.jpg"}})
이를 실행하면 위의 이미지가
![panda-bright]({{"/assets/img/imageProcessing/color-gray-panda-bright.jpg"}})
이와 같이 밝아진다.

#### 콘트라스트(contrast) 조절

콘트라스트 기능 구현만 밝기 조절과 다르고 다른 부분은 같다.

ImageProc.h, ImageProc.cpp 에 다음과 같은 contrast 조절 함수를 추가한다.
> ImageProc.h
{% highlight cpp %}
class ImageProc{
    ...

public:
    static void AdjustContrast(unsigned char* image,
        const int width, const int height, const int contrast );
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::AdjustContrast(unsigned char* image,
        const int width, const int height, const int contrast )
{
    // 255 보다 크면 255 0보다 작으면 0을 저장한다.
    // 밝기와 다른 유일한 부분은 (* contrast) 이다.
    for(int i=0; i<width*height; i++)
    {
        image[i*4 + 0] = static_cast<char>(min(255,(max(0,(image[i*4 + 0] * contrast)))));
        image[i*4 + 1] = static_cast<char>(min(255,(max(0,(image[i*4 + 1] * contrast)))));
        image[i*4 + 2] = static_cast<char>(min(255,(max(0,(image[i*4 + 2] * contrast)))));
    }
}
{% endhighlight%}

이제 contrast 조절 이벤트를 추가한다.

> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc::public CDocument
{
    ...
public :
    afx_msg void OnPixelProcessingAdjustcontrast();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
...
void OnPixelProcessingAdjustcontrast()
{
    ImageProc::AdjustContrast(m_Images[cur_index].image_color, 
        m_Images[cur_index].width, m_Images[cur_index].height, 3);

    // 작업완료된 이미지로 흑백 이미지를 만든다
    ImageProc::MergeChannels(m_Images[cur_index].image_color, 
       m_Images[cur_index].image_gray, m_Images[cur_index].width, m_Images[cur_index].height);
    
    // View 포인터를 얻어오고 이미지를 세팅한다.
    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();
    pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

    // 화면을 업데이트 한다.
    pView->OnInitialUpdate();
}
...
{% endhighlight %}

실행하면 다음과 그림과 같이 보인다.
![contrast-panda]({{"/assets/img/imageProcessing/contrast-panda.jpg"}})