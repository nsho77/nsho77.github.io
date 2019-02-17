---
layout: post
title: 방향키를 눌렀을 때 물체를 보는 각도를 바꿔보자2
date: 2018-08-13 13:00:00 +0900
description: EyeCoord 를 임의의 점을 중심으로 회전시켜보자2 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

이전에는 방향키 이벤트를 적용하기 위해 view 클래스에서 Renderer 클래스 포인터를 받아서 구현했다.

하지만 이와 같은 방식은 코드의 중복이 생겨 유지, 보수가 어려워지는 문제점이 있다. 

구조를 바꿔서 이해하기 쉽고 관리하기 쉬운코드를 작성해보자.

View 클래스에서 Doc 포인터의 메세지 함수를 불러오는 방식이다. 하지만 메세지 함수에는 파라미터를 전달 할 수 없다.

따라서 왼쪽, 오른쪽 방향키등의 파라미터를 직접 전달하는 것은 불가능하다. 

이를 해결하기 위해, 메시지 함수 호출을 도와주는 함수를 만들어 사용해보자.

먼저 View 클래스를 수정하자.

> VolumeRendererView.cpp
{% highlight cpp %}
void CVolumeRendererView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	CVolumeRendererDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	printf("MIP AnyDirection \n");

	switch (nChar)
	{
	case VK_LEFT: // 왼쪽 화살표키 눌러짐.
		printf("Hello Left Key\n");
        /// 기존에는 아래와 같이 했다.        
		//pDoc->OnMiprenderingAnydirection();
        /// Doc 클래스에 헬퍼함수를 만들어 사용하자.
		pDoc->DirKeyDownRendering(LEFT);

		printf("Hello Left Key end\n");
		break;
	case VK_RIGHT: // 오른쪽 화살표 키 눌러짐.
		printf("Hello Right Key\n");
		//pDoc->OnMiprenderingAnydirection();
		pDoc->DirKeyDownRendering(RIGHT);
		printf("Hello Right Key end\n");
		break;
	}
}
{% endhighlight %}

헬퍼함수에 전달할 매개변수를 보기 편하게 하기 위해 enum 을 선언하고 사용해보자.
> GlobalDefine.h
{% highlight cpp %}
enum{
    LEFT, RIGHT, UP, DOWN
}
{% endhighlight %}

{% highlight cpp %}
/// 아래 구문을 View 클래스, Doc 클래스, Renderer 클래스에 선언하자.
#include "GlobalDefine.h"
{% endhighlight %}

> VolumeRendererDoc.cpp
{% highlight cpp %}
void CVolumeRendererDoc::DirKeyDownRendering(int DirKey)
{
    m_dirKey = DirKey;
    OnMiprenderingAnydirection();
}

/// m_dirkey를 멤버변수로 Doc 클래스에 선언하고 클래스 생성시 초기화한다.
CVolumeRendererDoc::CVolumeRendererDoc
{
    m_dirKey = -1;
}


/// OnMiprenderingAnydirection 구현 부분을 m_dirKey 를 사용하도록 수정한다.
void CVolumeRendererDoc::OnMiprenderingAnydirection()
{
    ...
    m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height,m_DirKey);
    ...
}
{% endhighlight %}

> Renderer.cpp
{% highlight cpp %}
/// 눈 좌표를 멤버변수로 선언하고 이용한다.
Renderer::Renderer()
{
    m_eye_coord = {255.f/ 2.f, 255.f, 224.f / 2.f};
}

bool Renderer::RenderMIPAnyDirection(unsigned char* image,
	const int img_width, const int img_height, 
	int DirKey)
{
    int vol_width = m_pVolume->GetWidth();
    int vol_height = m_pVolume->GetHeight();
    int vol_depth = m_pVolume->GetDepth();

    /// 눈좌표, 업벡터, 센터 설정
    float3 up_vector = {0.f, 0.f, -1.f};
    float3 center_coord = {vol_width/2.f, vol_height/2.f, vol_depth/2.f};

    /// 눈좌표 rotate
	float angle = 0.f;
	switch (DirKey)
	{
	case LEFT:
		angle = +20.f;
		break;
	case RIGHT:
		angle = -20.f;
		break;
	default:
		break;
	}

	angle = angle * 3.141592f / 180.f;
	float cos_ = cosf(angle);
	float sin_ = sinf(angle);
	float x2 = center_coord.x * center_coord.x;
	float y2 = center_coord.y * center_coord.y;
	float z2 = center_coord.z * center_coord.z;
	float oneMinusCos = 1.f - cos_;


	float rotate_matrix[3][3] = {
		{cos_, (-1)*sin_, 0.f },
		{sin_, cos_ , 0.f },
		{0.f,0.f,1.f}
	};

    float rotate_eye[3] = { m_eye_coord.x, m_eye_coord.y, m_eye_coord.z };
	rotate_eye[0] -= center_coord.x;
	rotate_eye[1] -= center_coord.y;
	rotate_eye[2] -= center_coord.z;
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

	res_arr[0] += center_coord.x;
	res_arr[1] += center_coord.y;
	res_arr[2] += center_coord.z;

    m_eye_coord.x = res_arr[0]; m_eye_coord.y = res_arr[1]; m_eye_coord.z = res_arr[2];

    /// 뷰벡터 계산
	/// 연산자 오버로딩이 필요하다
	float3 view_vector = center_coord - m_eye_coord;
	//printf("view_vector :%f %f %f\n",view_vector.x, view_vector.y, view_vector.z);
	/// 구조체 안에 함수를 정의한다.
	view_vector.normalize();
	//printf("view_vector :%f %f %f\n", view_vector.x, view_vector.y, view_vector.z);


	/// x벡터 계산
	float3 x_vector = cross(view_vector, up_vector);
	//printf("x_vector :%f %f %f\n", x_vector.x, x_vector.y, x_vector.z);
	x_vector.normalize();
	//printf("x_vector :%f %f %f\n", x_vector.x, x_vector.y, x_vector.z);

	///y벡터 계산
	float3 y_vector = cross(view_vector, x_vector);
	y_vector.normalize();

	for (int i = 0; i < img_width; i++)
	{
		for (int j = 0; j < img_height; j++)
		{
			///시작 픽셀의 3차원 좌표 계산
			float3 cur_coord = m_eye_coord +
				x_vector * (i-img_width/2) + y_vector * (j-img_height/2);

			float t[2] = { 0.f, 0.f };
			GetRayBound(t, cur_coord, view_vector);

			unsigned char max_value = 0;
			for (float k = t[0]; k < t[1]; k += 1.f)
			{
				float3 adv_coord = cur_coord + view_vector * k;

				///진행 픽셀의 현재 위치가 볼륨 바운더리 안에 들어왔다면 
				if (adv_coord.x >= 0.f && adv_coord.x < vol_width-1 &&
					adv_coord.y >= 0.f && adv_coord.y < vol_height-1 &&
					adv_coord.z >= 0.f && adv_coord.z < vol_depth-1)
				{
					///해당 위치에서의 볼륨 복셀을 가져옴
					float voxel = 
						m_pVolume->GetVoxel(adv_coord.x, adv_coord.y, adv_coord.z);

					///맥스값 비교
					max_value = __max(max_value, voxel);
				}

				///뷰벡터 방향으로 한칸 전진 (view_vector는 normalize를 했으므로, 1 만큼 전진함)
				//cur_coord = cur_coord + view_vector;
			}

			///마지막 값 이미지에 대입
			image[img_height * j + i] = static_cast<unsigned char>(max_value);
		}
	}
    return true;
}
{% endhighlight %}