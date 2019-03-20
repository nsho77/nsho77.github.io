---
layout: post
title: 어떤 방향에서도 볼 수 있는 MIP 를 만들어보자.
date: 2018-07-18 10:45:00 +0900
description: Volume 으로 MIP 를 만들어보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

이전 MIP 는 X, Y, Z 방향에서 본 화면 이었다.

이번에는 사용자가 어떤 좌표에서 보고 싶은지 정하면 해당 좌표에서 바라본 MIP를 만들어보자.

개발 순서는 다음과 같다.
1. view vector 를 구한다.
2. x vector 를 구한다.
3. y vector 를 구한다.
4. 구한 vector 평면에서 view vector 방향으로 전진하면서 MIP를 구한다.

먼저 이벤트 처리기 구현하자
> VolumeRendererDoc.cpp
{% highlight cpp %}
void CVolumeRendererDoc::OnMiprenderingAnydirection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int img_width = 256;
	int img_height = 256;

	shared_ptr<unsigned char> image =
		shared_ptr<unsigned char>(new unsigned char[img_width*img_height]);
	memset(image.get(), 0, sizeof(unsigned char)*img_width*img_height);

	m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height);

	CVolumeRendererView* pView =
		(CVolumeRendererView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(image.get(), img_width, img_height, 1);

	pView->OnInitialUpdate();
}
{% endhighlight %}

핵심 함수인 RenderMIPAnyDirection 을 구현하자. 해당 함수에서 3차원 좌표가 자주 사용되므로
이를 구조체로 선언하여 간편하게 쓰도록 하자.

> Renderer.h
{% highlight cpp %}
#include "Volume.h"
...
struct float3
{
    float x;
    float y;
    float z;

    float3()
    {
        x = 0.f; y = 0.f; z = 0.f;
    };

    float3(float _x, float _y, float _z)
    {
        x = _x; y = _y; z = _z;
    };
}
{% endhighlight %}

> Renderer.cpp
{% highlight cpp %}
bool Renderer::RenderMIPAnyDirection(unsigned char* image,
	const int img_width, const int img_height)
{
    int vol_width = p_mVolume -> GetWidth();
    int vol_height = p_mVolume -> GetHeight();
    int vol_depth = p_mVolume -> GetDepth();

    // 눈, 센터, 업 좌표 설정
    float3 eye_coord = float3(100.f, 200.f, 300.f);
    float3 center_coord = float3(vol_width/2.f, vol_height/2.f, vol_depth/2.f);
    float3 up_vector = float3(0.f, 0.f, -1.f);

    // view vector 계산
    float3 view_vector = center_coord - eye_coord;

    // x vector 계산
    float3 x_vector = 외적(view_vector, up_vector);
    x_vector.normalize();


    // y vector 계산
    float3 y_vector.외적(view_vector, x_vector);
    y_vector.normalize();

    for(int j=0; j<img_height; j++)
    {
        for(int i=0; i<img_width; i++)
        {
            float3 cur_coord = eye_coord + x_vector*(i-img_width/2) + y_vector*(j-img_height/2);

            unsigned char max_value = 0;
            for(int k=0; k<1000; k++)
            {
                if( cur_coord.x >=0 && cur_coord.x < vol_width
                && cur_coord.y >= 0 && cur_coord.y < vol_width
                && cur_coord.z >= 0 && cur_coord.z < vol_width )
                {
                    max_value = __max(max_value, m_pVolume->GetVoxel(cur_coord.x,cur_cood.y,cur_coord.z));
                }
                // view_vector 만큼 전진한다.
                cur_coord += view_vector;                
            }
            image[img_width*j + i] = max_value;
        }
    }
    
}

{% endhighlight %}


이제 정규화와 외적 함수를 구현하자. 정규화는 normalize, 외적은 cross 라는 이름으로 사용할 것이다.

정규화함수는 float3 구조체 내에서 선언 후 사용할 것이다. 연산자 오버로딩도 구현해놓자.
> Renderer.h
{% highlight cpp %}
#include <math.h>
struct float3
{
    ...
    // 연산자 오버로딩
    float3 operator- (float3 s2)
	{
		float3 res = float3();
		res.x = x - s2.x;
		res.y = y - s2.y;
		res.z = z - s2.z;
		return res;
	}

	float3 operator+ (float3 s2)
	{
		float3 res = float3();
		res.x = x + s2.x;
		res.y = y + s2.y;
		res.z = z + s2.z;
		return res;
	}

	float3 operator* (int i)
	{
		float3 res = float3();
		res.x = x * i;
		res.y = y * i;
		res.z = z * i;
		return res;
	}

    // 정규화
    void normalize()
    {
        float dist = sqrt(x*x + y*y + z*z);
        if(dist > 0)
            x = x / dist; y= y/dist; z= z/dist;
    }
}
{% endhighlight %} 

외적구현
> Renderer.cpp
{% highlight cpp %}
float3 cross(float3 v1, float3 v2)
{
    float res_x = v1.y*v2.z - v1.z*v2.y;
    float res_y = v1.z*v2.x - v1.x*v2.z;
    float res_z = v1.x*v2.y - v1.y*v2.x;

    float3 res = float3(res_x,res_y,res_z);
    return res;
}
{% endhighlight %}

실행하면 다음과 같이 나온다.

![MIP-any]({{"/assets/img/Volume/MIP-any.png"}})