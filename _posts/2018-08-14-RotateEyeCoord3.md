---
layout: post
title: 방향키를 눌렀을 때 물체를 보는 각도를 바꿔보자3
date: 2018-08-14 10:00:00 +0900
description: EyeCoord 를 임의의 점을 중심으로 회전시켜보자3 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

이전까지 방향키를 눌렀을 때 MIP 를 회전하면서 보여주는 기능을 만들었다. 여기에는 문제가 있다.

아무런 동작을 취하지 않고 방향키만 눌렀을 때 MIP 가 보여진다. 

이 버그를 MIP 를 실행했을 때, 방향키가 눌러지면 회전하는 것으로 수정하자.

> VolumeRendererDoc.cpp
{% highlight cpp %}
void CVolumeRendererDoc::DirKeyDownRendering(int DirKey)
{
	m_DirKey = DirKey;

    /// Renderer 클래스에 CurMode 변수를 추가할 것이다.
    /// 값에 따라서 동작을 다르게 한다.
	int CurMode = m_pRenderer->GetCurMode();
	switch (CurMode)
	{
	case SLICE:
		break;
	case MIP:
		OnMiprenderingAnydirection();
		break;
	case VR:
		break;
	default:
		break;
	}

}

...
/// MIP 뿐 아니라 모든 기능에 Renderer 클래스의
/// 새로 추가한 멤버변수를 세팅하는 로직을 만든다.
void CVolumeRendererDoc::OnMiprenderingAnydirection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int img_width = 256;
	int img_height = 256;

	printf("MIP AnyDirection \n");

	shared_ptr<unsigned char> image =
		shared_ptr<unsigned char>(new unsigned char[img_width*img_height]);
	memset(image.get(), 0, sizeof(unsigned char)*img_width*img_height);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&BeginTime);

    /// MIP 이므로 MIP로 세팅한다.
	m_pRenderer->SetCurMode(MIP);
	m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height,m_DirKey);
	

	QueryPerformanceCounter(&Endtime);
	int elapsed = Endtime.QuadPart - BeginTime.QuadPart;
	double duringtime = (double)elapsed / (double)Frequency.QuadPart;

	printf("MIP AnyDirection time : %f\n", duringtime);

	CVolumeRendererView* pView =
		(CVolumeRendererView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(image.get(), img_width, img_height, 1);

	pView->OnInitialUpdate();

	printf("MIP AnyDirection End \n");
}
{% endhighlight %}


> Renderer.cpp
{% highlight cpp %}
/// enum, m_CurMode 을 추가한다.
enum{
    SLICE, MIP, VR
}

class ...
{
    ...
private:
    int m_CurMode;
public:
    int GetCurMode();
    void SetCurMode(int CurMode);
}

/// 새로만든 변수를 초기화하고 기능을 구현한다.
Renderer::Renderer
{
    ...
    m_CurMode = -1;
}

Renderer::Renderer(unsigned char* volume, int width, int height, int depth)
{
    ...
    m_CurMode = -1;
}

int Renderer::GetCurMode()
{
    retunr m_CurMode;
}

void Renderer::SetCurMode(int CurMode)
{
    m_CurMode = CurMode;
}
{% endhighlight %}