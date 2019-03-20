---
layout: post
title: 방향키를 눌렀을 때 물체를 보는 각도를 바꿔보자
date: 2018-07-27 09:00:00 +0900
description: EyeCoord 를 임의의 점을 중심으로 회전시켜보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

방향키를 눌렀을 때 보는 점(시점)의 좌표를 특정 각도 만큼 회전시켜 보자.

구현하기 위해 먼저 키보드 입력 이벤트를 어떻게 처리하는지 알아보자.

클래스 위저드에서 메시지 선택을 WM_KEYDOWN을 선택하여 함수를 추가한다. 아래와 같은 함수가 추가된다.
{% highlight cpp %}
void CVolumeRendererView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CView::OnKeyDown(nChar, nRepCnt, nFlags);
}
{% endhighlight %}

이를 다음과 같이 방향키에서만 작동하도록 수정한다.
{% highlight cpp %}
void CVolumeRendererView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	switch(nChar)
    {
        case VK_LEFT: /// 왼쪽 화살표키
            printf("Left Key \n");

            printf("Left Key End \n");
            break;
        case VK_RIGHT: /// 오른쪽 화살표키
            printf("Left Key \n");

            printf("Left Key End \n");
            break;
    }
}
{% endhighlight %}

왼쪽키 오른쪽 키를 눌렀을 때 MIP ANY DIRECTION 함수를 호출 하게 만들어보고 잘 동작하는지 확인해보자.
{% highlight cpp %}
void CVolumeRendererView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
    CVolumeRendererDoc* pDoc = GetDocument();
    ASSERT_VALID(pDOc);
    if(!pDoc) return;

    int img_width = 256;
    int img_height = 256;

    printf("MIP AnyDirection \n");
    shared_ptr<unsigned char> image = 
        shared_ptr<unsigned char>(new unsigned char[img_width*img_height]);

    // Renderer 포인터를 가져온다.
    // Doc 클래스에 Renderer 포인터를 가져오는 함수를 만든다.
    // 이후 아래와 같이 함수를 이용해 Renderer 클래스를 가리키는 포인터를 가져온다.
    //shared_ptr<Renderer> m_pRenderer = pDoc->Getm_pRenderer();

	switch(nChar)
    {
        case VK_LEFT: /// 왼쪽 화살표키
            printf("Left Key \n");
            m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height);
            printf("Left Key End \n");
            break;
        case VK_RIGHT: /// 오른쪽 화살표키
            printf("Right Key \n");
            m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height);
            printf("Right Key End \n");
            break;
    }
}
{% endhighlight %}


Doc 에서 Renderer 클래스 포인터를 가져올 수 있는 함수를 만든다.
> VolumeRendererDoc.cpp
{% highlight cpp %}
shared_ptr<Renderer> VolumeRenderer::Getm_pRenderer()
{
    return m_pRenderer;
}
{% endhighlight %}

이제 키를 누르면 시점의 좌표를 변환시켜보자. 이를 구현하기 위해 
변환행렬을 이용한다.

이는 원점을 중심으로 시점의 좌표를 반시계 방향으로 a 각도만큼 이동시키는 행렬이다.

> Renderer.cpp
{% highlight cpp %}
/// 시점을 얼마큼 회전시킬지 알수 있는 각도를 매개변수로 받는다.
bool Renderer::RenderMIPAnyDirection(unsigned char* image,
	const int img_width, const int img_height,
	float angle)
{
    ...

/// 눈좌표 rotate
	angle = angle * 3.141592f / 180.f;
	float cos_ = cosf(angle);
	float sin_ = sinf(angle);
	float x2 = center_coord.x * center_coord.x;
	float y2 = center_coord.y * center_coord.y;
	float z2 = center_coord.z * center_coord.z;
	float oneMinusCos = 1.f - cos_;

	float rotate_matrix[3][3] = {
		{ cos_, (-1)*sin_, 0.f },
		{ sin_,     cos_ , 0.f },
		{  0.f,       0.f, 1.f }
	};

    /// 눈 좌표에 위에서 선언한 회전행렬을 곱하여 시점을 회전시킨다.

	float rotate_eye[3] = { eye_coord.x, eye_coord.y, eye_coord.z };
	float res_arr[3] = { 0.f };
	for (int i = 0; i < 3; i++)
	{
		float res = 0.f;
		for (int j = 0; j < 3; j++)
		{
			res = res + rotate_eye[j] * rotate_matrix[i][j];
		}
		res_arr[i] = res;
	}

	eye_coord.x = res_arr[0]; eye_coord.y = res_arr[1]; eye_coord.z = res_arr[2];

    ...

}
{% endhighlight %}

이제 View 를 아래와 같이 다시 수정하고 실행하면 원점을 중심으로 시점이 회전하는 것을 볼 수 있다.
> VolumeRendererView.cpp
{% highlight cpp %}
void CVolumeRendererView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
    ...
	

switch (nChar)
	{
	case VK_LEFT: // 왼쪽 화살표키 눌러짐.
		printf("Hello Left Key\n");
		total_angle -= 25.f;
		m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height, total_angle);

		printf("Hello Left Key end\n");
		break;
	case VK_RIGHT: // 오른쪽 화살표 키 눌러짐.
		printf("Hello Right Key\n");
		total_angle += 25.f;
		m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height, total_angle);

		printf("Hello Right Key end\n");
		break;
	}
    ...

}
{% endhighlight %}