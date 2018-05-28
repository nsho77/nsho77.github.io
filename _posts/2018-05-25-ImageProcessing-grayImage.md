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

    for(int i=0; i<width; i++)
    {
		for(int j=0; j<height; j++)
		{
			m_Image[(m_ImgWidth*j+i)*4 + 0] = image[(width*j+i)*4 + 0];
        	m_Image[(m_ImgWidth*j+i)*4 + 1] = image[(width*j+i)*4 + 1];
        	m_Image[(m_ImgWidth*j+i)*4 + 2] = image[(width*j+i)*4 + 2];
		}
    }

	// color image 다음에 gray image 가 나오므로 + width 를 한다.
    for(int i=0; i<width; i++)
    {
		for(int j=0; j<height; j++)
		{
			m_Image[(m_ImgWidth*j+i+width)*4 + 0] = image_gray[width*j+i];
			m_Image[i*4 + width + 1] = image_gray[width*j+i];
			m_Image[i*4 + width + 2] = image_gray[width*j+i];
		}
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
gray image 도 데이터 저장을 해야하므로 IMAGE 구조체를 먼저 수정하자.
{% highlight cpp %}
// ImageProcessingDoc.h 임.
#include <vector>
using namespace std;

struct IMAGE
{
	unsigned char* image_color;
	unsigned char* image_gray;
	int width;
	int height;
	int pixel_byte;
	// gray image 가 binary 처리 되었는가?
	bool bGrayBinary;

	IMAGE(unsigned char* _color, unsigned char* _gray, int _width,
		int _height, int _pixel_byte)
	{
		image_color = nullptr;
		image_gray = nullptr;
		width = 0;
		height = 0;
		pixel_byte = 0;
		bGrayBinary = false;
	}

	IMAGE(unsigned char* _color, unsigned char* _gray, int _width,
		int _height, int _pixel_byte)
	{
		image_color = _color;
		image_gray = _gray;
		width = _width;
		height = _height;
		pixel_byte = _pixel_byte;
		bGrayBinary = false;
	}
}

// 클래스에 이미지들을 저장할 백터배열과 인덱스를 만든다.
class CImageProcessingDoc() : public CDocument
{
	...
public :
	vector <IMAGE> m_Images;
	int cur_index;
}
{% endhighlight %}

이제 File Open 했을 때 이벤트 처리 기능을 구현해보자.
{% highlight cpp %}
void CImageProcessingDoc::OnFileOpen()
{
	TCHAR szFilter[] = _T("Image (*.BMP, *.GIF, *.JPG, *.PNG) | *.BMP;*.GIF;*.JPG;*.PNG; |
		All Files(*.*)|*.*||");
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);

	if(IDOK == dlg.DoModal())
	{
		CString strPathName =dlg.GetPathName();

		CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

		unsigned char* img = nullptr;
		int width, height, byte = 0;

		if(pView->ReadImageFile(strPathName, img, width, height, byte))
		{
			printf("open success\n");

			// gray image 버퍼를 만들고 이미지를 저장한다.
			unsigned char* image_gray = new unsigned char[width*height];
			ImageProc::MergeChannels(img, image_gray, width, height);

			// IMAGE 변수를 만들고 데이터를 저장한뒤 배열에 저장한다.
			IMAGE image_info = IMAGE(img, image_gray, width, height, 4);
			m_Images.push_back(image_info);
			cur_index = static_cast<int>(m_Images.size() -1) ;

			// 현재 이미지를 화면에 뿌린다.
			pView->setDrawImage(img, image_gray, m_Images[cur_index].width, m_Images[cur_index].height);

			// 화면을 update 한다.
			pView->OnInitialUpdate();

		}
	}
}
{% endhighlight %}