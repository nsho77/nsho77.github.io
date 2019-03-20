---
layout: post
title: mfc-imageProcessing 바이너리 이미지를 만들어보자
date: 2018-05-29 17:00:00 +0900
description: mfc imageprocessing 다양한 이미지 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
categories: [Image Processing]
---

[이전](../various-event-image)까지 밝기와 contrast 를 조절하는 작업을 했다.

# 픽셀 단위 작업

* 밝기 조절
* 콘트라스트 조절
* 흑백이미지를 바이너리화

# 영역 단위 작업

* 이진 이미지 침식
* 이진 이미지 팽창

이제 흑백이미지를 바이너리화 하는 작업을 해보자.

#### 흑백이미지 바이너리화
> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void Binarization(unsigned char* image_gray,
        const int width, const int height, const unsigned char threshold);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
// threshold 값보다 작으면 0 크면 255를 만드는 함수이다.
void ImageProc::Binarization(unsigned char* image_gray,
    const int width, const int height, const unsigned char threshold)
{
    for(int i=0; i<width*height; i++)
    {
        if(image_gray[i] < threshold)
            image_gray[i] = 0;
        else
            image_gray[i] = 255;
    }
}
{% endhighlight %}

> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc : public CDocument
{
    ...
public:
    afx_msg void OnPixelprocessingBinarization();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnPixelprocessingBinarization()
{
    ImageProc::Binarization(m_Images[cur_index].image_gray,
        m_Images[cur_index].width, m_Images[cur_index].height, 127);

    CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 다음과 같은 화면이 나온다.
![binary-panda]({{"/assets/img/imageProcessing/binary-panda.jpg"}})