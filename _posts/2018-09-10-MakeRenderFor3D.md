---
layout: post
title: 3D 이미지를 화면에 뿌려보자 - 2
date: 2018-09-10 09:00:00 +0900
description: 광선을 추적하면서 렌더해보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

ray-casting (광선 추척법)을 이용하여 3D volume 을 rendering 해보자.


> Renderer.cpp
{% highlight cpp %}
bool Renderer::RenderVRAnyDirection(unsigned char* image,
    const int img_width, const int img_height, int DirKey)
{
    if(m_CurMode != VR) return false;

    int vol_width = m_pVolume->GetWidth();
    int vol_height = m_pVolume->GetHeight();
    int vol_depth = m_pVolume->GetDepth();

    /// 업벡터, 볼륨 센터 설정
    float3 up_vector = float3(0.f, 0.f, -1.f);
    float3 center_coord = float3(vol_width / 2.f, vol_height / 2.f, vol_depth / 2.f);

    /// 눈 좌표 회전
    float angle = 0.f;

    switch(DirKey)
    {
    case LEFT:
        angle = +10.f;
        break;
    case RIGHT:
        angle = -10.f;
        break;
    default:
        break;
    }

    angle = angle * 3.141592f / 180.f;
    float cos_ = cosf(angle);
    float sin_ = sinf(angle);
    
    float rotate_matrix[3][3] = {
        {cos_, (-1)*sin_, 0.f},
        {sin_,      cos_, 0.f},
        { 0.f,       0.f, 1.f}
    }

    float rotate_eye[3] = { m_eye_coord.x, m_eye_coord.y, m_eye_coord.z};
    rotate_eye[0] -= center_coord.x;
    rotate_eye[1] -= center_coord.y;
    rotate_eye[2] -= center_coord.z;

    float res_arr[3] = { 0.f };

    for(int i=0; i<3; i++)
    {
        float res = 0.f;
        for(int j=0; j<3; j++)
            res = res + rotate_eye[j] * rotate_matrix[i][j];
        
        res_arr[i] = res;
    }

    res_arr[0] += center_coord.x;
    res_arr[1] += center_coord.y;
    res_arr[2] += center_coord.z;

    m_eye_coord.x = res_arr[0];
    m_eye_coord.y = res_arr[1];
    m_eye_coord.z = res_arr[2];

    /// 뷰 벡터 계산
    float3 view_vector = center_coord - m_eye_coord;
    view_vector.normalize();

    /// x 벡터 계산
    float3 x_vector = cross(view_vector, up_vector);
    x_vector.normalize();

    /// y 벡터 계산
    float3 y_vector = cross(view_vector, x_vector);
    y_vector.normalize();

    for(int j=0; j< img_height; j++)
    {
        for(int i=0; i<img_width; i++)
        {
            /// 시작좌표 계산
            float3 cur_coord = m_eye_coord + x_vector*(i - img_width/2) + y_vector*(j - img_height/2);

            float t[2] = { 0.f };
            GetRayBound(t, cur_coord, view_vector);

            float color[3] = { 0.f };
            float alpha = 0.f;

            for(float k = t[0]; k < t[1]; t+=1.f)
            {
                float3 adv_coord = cur_coord + view_vector * k;
                if(adv_coord.x >= 0.f && adv_coord.x < vol_width   &&
                   adv_coord.y >= 0.f && adv_coord.y < vol_height  &&
                   adv_coord.z > =0.f && adv_coord.z < vol_depth )
                   {
                       int intensity = m_pVolume->GetVoxel(static_cast<int>(adv_coord.x),static_cast<int>(adv_coord.y), static_cast<int>(adv_coord.z));
                       float cur_blue = m_pTF->GetPalleteCValue(0, intensity);
                       float cur_green = m_pTF->GetPalleteCValue(1, intensity);
                       float cur_red = m_pTF->GetPalleteCValue(2, intensity);
                       float cur_alpha = m_pTF->GetPalleteAValue(intensity);

                       color[0] += (1.f - alpha)* cur_blue * cur_alpha;
                       color[1] += (1.f -alpha) * cur_green * cur_alpha;
                       color[2] += (1.f -alpha) * cur_red * cur_alpha;
                       alpha += (1.f - alpha) * cur_alpha;

                       if (alpha > 0.95f) break;
                   }
            }
            image[(img_width*j + i) * 3 + 0] = color[0];
            image[(img_width*j + i) * 3 + 1] = color[1];
            image[(img_width*j + i) * 3 + 2] = color[2];
        }
    }
    return true;
}
{% endhighlight %}

![bighead-1dpallet]({{"/assets/img/Volume/bighead-1dpallet.png"}})