---
layout: post
title: mfc-imageProcessing next 이벤트를 달아보자
date: 2018-05-25 08:45:00 +0900
description: mfc imageprocessing next 이벤트를 달자 # Add post description (optional)
img:  # Add image post (optional)
---

기존 ImageProcessing [작업](../mfcImageProcessing)은 파일 다이얼로그에서 이미지 파일을 읽은 뒤 해당 함수에서 이미지를 세팅하는 함수를 호출하는 구조였다. (아래코드 참조)
{% highlight cpp %}
bool CImageProcessingView::ReadImageFile(CString filename,
	unsigned char*& output, int& width, int& height, int& byte)
{
    ...

    SetDrawImage(output, bm.bmWidth, bm.bmHeight, bm.bmBitsPixel/8);
}
{% endhighlight %}

이벤트를 달기 앞서, SetDrawImage를 함수 내에서 호출 하지 말고 독립적으로 호출 시켜 사용할 수 있도록 구조를 조금 바꿔보자.
기존 코드는 아래와 같다.
{% highlight cpp %}
bool CImageProcessingView::ReadImageFile(CString filename,
    unsigned char*& output, int& width, int& height, int& byte)
{
    CBitmap Bitmap;
    Bitmap.m_hObject = 
        (HBITMAP)::LoadImage(NULL, filename, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
    
    if(!Bitmap.m_hObject) return false;

    BITMAP bm;
    Bitmap.GetBitmap(&bm);

    DWORD dwCount = bm.bmWidthBytes*bm.bmHeight;

    output = new unsigend char[dwCount];
    DWORD dwRead = Bitmap.GetBitmapBits(dwCount, output);

    SetDrawImage(output, bm.bmWidth, bm.bmHeight, bm.bmBitsPixel/8);

    width = bm.bmWidth;
    height = bm.bmHeight;
    byte = bm.bmBitsPixel / 8;
}
{% endhighlight %}

이를 아래와 같이 수정한다

{% highlight cpp %}
bool CImageProcessingView::ReadImageFile(CString filename,
	unsigned char*& output, int& width, int& height, int& byte)
{
	CBitmap Bitmap;
	Bitmap.m_hObject = 
		(HBITMAP)::LoadImage(NULL, filename, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
	if (!Bitmap.m_hObject) return false;

	BITMAP bm;
	Bitmap.GetBitmap(&bm);

	DWORD dwCount = bm.bmWidthBytes*bm.bmHeight;

	output = new unsigned char[dwCount];
	DWORD dwRead = Bitmap.GetBitmapBits(dwCount, output);
	
	width = bm.bmWidth;
	height = bm.bmHeight;
	byte = bm.bmBitsPixel / 8;

	return true;
}
{% endhighlight %}

달라진 점은 SetDrawImage를 호출하지 않게 바꾼 것 뿐이다.

이제 간단히 next 아이콘을 클릭 하면 다음 이미지가 그려지는 기능을 추가해보자.
ImageProcessingDoc.h, ImageProcessingDoc.cpp 에 다음과 같은 이벤트처리기를 만든다.

> ImageProcessingDoc.h
{% highlight cpp %}
...
public:
    afx_msg void OnNextImage();
...
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
void OnNextImage()
{
    cur_index = (cur_index+1) % m_Images.size();

    CImageProcessingView* pView =
		(CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();
    
    pView->setDrawImage(m_Images[cur_index].image, m_Images[cur_index].width, m_Images[cur_index].height, m_Images[cur_index].byte);
    pView->OnInitialUpdate();
}
{% endhighlight %}