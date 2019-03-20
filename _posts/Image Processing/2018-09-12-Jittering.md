---
layout: post
title: 3D 이미지 나뭇결무늬를 제거하자 - 2
date: 2018-09-12 21:00:00 +0900
description: jittering # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

jittering 이라는 방법으로 나뭇결 무늬를 제거해보자.

시점의 좌표를 랜덤하게 흔들어서 나뭇결무늬를 없애는 방법이다.

> Renderer.cpp
{% highlight cpp %}
#include <time.h>

bool Renderer::RenderVRAnyDirection(unsigned char* image,
	const int img_width, const int img_height, int DirKey)
{
...
    srand(time(NULL));
    for(int j=0; j<img_height; j++)
    {
        for(int i=0; i<img_width; i++)
        {
            float3 cur_coord = m_eye_coord + x_vector*(i - img_width/2) + y_vector(j - img_height/2);

            float t[2] = {0.f};
            GetRayBound(t, cur_coord, view_vector);

            /// Jittering 시작점을 흔들어서 나뭇결 무늬를 감춘다.
            float random = static_cast<float>( (rand() % 1000) ) / 1000.f ;
            cur_coord.x += random * view_vector.x;
            cur_coord.y += random * view_vector.y;
            cur_coord.z += random * view_vector.z;
            ...

        }
    }
}
{% endhighlight %}


jittering 적용한 영상은 아래와 같이 보인다.
![bighead-zittering]({{"/assets/img/Volume/bighead-zittering.png"}})