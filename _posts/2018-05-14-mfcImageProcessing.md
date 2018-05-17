---
layout: post
title: mfc-imageProcessing 을 훑어보자~!
date: 2018-05-14 19:20:00 +0900
description: mfc imageprocessing 하나하나 풀어보자 # Add post description (optional)
img:  # Add image post (optional)
---

mfc 로 프로젝트를 생성하면 다음과 같은 소스파일이 생성된다.
* ImageProcessing.cpp
* ImageProcessingDoc.cpp
* ImageProcessingView.cpp
* FileView.cpp
* MainFrm.cpp
...

이중에서 가장 중요한 소스는 위부터 3개다. 위 3개에 나머지 소스가 종속되어있다고 보면 된다.

ImagaProcessing.cpp 은 전체적인 흐름을 담당하고 ImageProcessingDoc.cpp 은 메뉴, 문서 등을 담당하며 ImageProcessingView.cpp 은 이미지가 보이는 화면을 담당한다.


먼저 ImageProcessingView.h를 보자 ImageProcessingView.cpp 에서 정의할 클래스의 변수, 메서드를 볼 수 있다.

{% highlight cpp %}
class CImageProcessingView : public CView
{
    ...
}
{% endhighlight %}

class 선언으로 class를 만들 수 있다. ':' 는 CView를 상속받는다는 의미다.

{% highlight cpp %}
class CImageProcessingView : public CView
{
protected:
    CImageProcessingView();
    DECLARE_DYNCREATE(CImageProcessingView);

private:
    ...

public:
    ...
}
{% endhighlight %}

protected, private, public 등의 접근 제한자로 상황에 알맞은 변수, 함수를 선언할 수 있다.

> 질문 : protected 안에 CImageProcessingView()는 무엇일까??

{% highlight cpp %}
CImageProcessingView::CImageProcessingView()
{
    m_ImgWidth = 0;
    m_ImgHeight = 0;
    m_Image = nullptr;
}

CImageProcessingView::~CImageProcessingView()
{
    if(m_Image)
        delete[] m_Image;
}
{% endhighlight %}

클래스와 이름이 같고 함수의 반환형이 표시되지 않은 함수는 생성자이다.
생성자 앞에 ~ 표시가 있으면 소멸자이다.

{% highlight cpp %}
void CImageProcessingView::OnDraw(CDC* pDC)
{
    CImageProcessingDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);
    if(!pDoc)
        return;

    // TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
    if(m_Image)
    {
        CDC MemDC;
        BITMAP bmpInfo;

        // 화면 DC와 호환되는 메모리 DC를 생성
        MemDC.CreateCompatibleDC(pDC);

        // 비트맵 리소스 로딩
        CBitmap cBitmap;
        cBitmap.CreateBitmap(m_ImgWidth, m_ImgHeight, 1, 32, m_Image);
        CBitmap* pOldBmp = NULL;

        // 로딩된 비트맵 정보 확인
        cBitmap.GetBitmap(&bmpInfo);
        // 정보 출력
        printf("view image width %d, height %d\n",bmpInfo.bmWidth,bmpInfo.bmHeight);

        // 메모리 DC에 선택
        pOldBmp = MemDC.SelectObject(&cBitmap);

        // 메모리 DC에 들어 있는 비트맵을 화면 DC로 복사하여 출력
        pDC->BitBlt(0,0,bmpInfo.bmWidth, bmpInfo.bmHeihgt, &MemDC, 0, 0, SRCCOPY);
    }
}
{% endhighlight %}

> 위의 코드는 화면에 그리기 위한 필요코드이다.

이미지 파일을 불러와 화면에 띄우는 과정은 다음과 같다.<br />

- Doc 에서 View의 함수에 접근하여 띄운다.
- Doc 에서 넘겨받은 정보로 이미지 정보를 셋팅한다.
- 화면에 이미지를 그린다.

화면에 그림을 그리기 위해 데이터를 셋팅하는 로직을 봐보자.

{% highlight cpp %}
// 먼저 ImageProcessingView.h 에 함수를 선언한다.
public:
    void SetDrawImage(unsigned char* image, const int width, const int height, const int byte);

// 이미지 배열, 사이즈, 한 픽셀에 몇 바이트인지 받는다.
// ImageProcessingView.cpp 에 함수를 정의한다.
void CImageProcessingView::SetDrawImage(unsigned char* image, const int width, const int height, const int byte)
{
    if(byte <= 0) return;

    // 이미지의 사이즈로 그림을 그린다.
    m_ImageWidth = width;
    m_ImageHeight = height;

    // 이미 그려진 이미지가 있다면 지운다.
    if(m_Image)
        delete[] m_Image;

    // 이미지를 그리기 위한 새로운 메모리를 할당한다.
    // mfc 는 1픽셀당 4바이트가 있으므로 다음과 같이 할당.
    m_Image = new unsigned char[width * height * 4];

    // 그리려는 이미지의 픽셀당 바이트 수에 따라
    // 알맞게 할당해준다.
    if(byte > 1)
    {
        // image는 width*height 만큼의 픽셀이고
        // 1픽셀은 byte 만큼을 가진다.
        for(int i=0; i<width*height; i++)
        {
            m_Image[i*4+0] = image[i*byte+0];
            m_Image[i*4+1] = image[i*byte+1];
            m_Image[i*4+2] = image[i*byte+2];
        }
    }
    else
    {
        // 흑백 image는 1픽셀에 1byte이다.
        for(int i=0; i< width*height; i++)
        {
            m_Image[i*4+0] = image[i];
            m_Image[i*4+1] = image[i];
            m_Image[i*4+2] = image[i];
        }
    }
}
{% endhighlight %}


Doc에서 파일을 읽어와 image 정보 셋팅 함수에 인자를 전달하는 함수를 살펴보자
{% highlight cpp %}
bool CImageProcessingView::ReadImageFile(CString filename,
    unsigned char*& output, int& width, int& height, int& byte)
{
    // CBitmap 구조체가 하는 일은 무엇?
    // LoadImage() 가 하는 역할은 무엇?
    CBitmap Bitmap;
    Bitmap.m_hObject = 
        (HBITMAP)::LoadImage(NULL, filename, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
    
    if(!Bitmap.m_hObject) return false;

    // BITMAP 구조체가 하는 일은 무엇?
    BITMAP bm;
    Bitmap.GetBitmap(&bm);

    // DWORD 구조체의 역할은 무엇?
    DWORD dwCount = bm.bmWidthBytes*bm.bmHeight;

    // 메모리 할당
    output = new unsigend char[dwCount];

    //GetBitmapBits() 가 하는 역할은? output 공간에 dwCount 만큼 bitmap을 생성
    DWORD dwRead = Bitmap.GetBitmapBits(dwCount, output);

    // output의 그림을 컴퓨터 화면에 띄우기 위해 세팅한다.
    // bmBitsPixel / 8 이유는? byte 가 아니라 bit로 표시돼서.
    SetDrawImage(output, bm.bmWidth, bm.bmHeight, bm.bmBitsPixel/8);

    // 밖으로 빼낼 정보를 세팅한다.
    width = bm.bmWidth;
    height = bm.bmHeight;
    byte = bm.bmBitsPixel / 8;
}
{% endhighlight %}

