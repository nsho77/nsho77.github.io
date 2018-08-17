---
layout: post
title: 3D 이미지를 화면에 뿌려보자 - 1
date: 2018-08-14 10:00:00 +0900
description: EyeCoord 를 임의의 점을 중심으로 회전시켜보자3 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
---

지금까지는 MIP 를 이용하여 3D 이미지를 화면에 뿌려보았다. 

가장 높은 Intensity 를 갖는 값만을 화면에 보여주었는데 이제부터는 사용자가 원하는 Intensity 범위를 받고 해당 Intensity 의 이미지에 색깔을 입혀 출력하는 기능을 구현해보자.

그러기 위해서 TransferFunction Class 를 정의한다. TransferFunction 클래스는 보고자 하는 Intensity 의 범위를 입력받고

해당 Intensity 에 색깔을 갖고 있는 팔레트를 만드는 역할을 한다.

> TransferFunction.h
{% highlight cpp %}
#include <memory>
using namespace std;
#define MAX_INTENSITY 256
class TransferFunction
{
private:
    shared_ptr<float>m_PalleteC[3];
    shared_ptr<float>m_PalleteA;
public:
    TransferFunction();
    TransferFunction(int start_color[3], int end_color[3], int start_alpha, int end_alpha);
    ~TransferFunction();
    void SetColorPallete(int start_color[3], int end_color[3]);
    void SetAlphaPallete(int start_alpha, int end_alpha);
    float GetPalleteCValue(int color, int intensity);
    float GetPalleteAValue(int intensity);
}
{% endhighlight %}


> TransferFunction.cpp
{% highlight cpp %}
#include "stdafx.h"
#include "TransferFunction.h"


TransferFunction::TransferFunction()
{
	m_PaletteC[0] = nullptr;
	m_PaletteC[1] = nullptr;
	m_PaletteC[2] = nullptr;
	m_PaletteA = nullptr;
}

TransferFunction::TransferFunction(int color_start[3], int color_end[3], 
	int alpha_start, int alpha_end)
{
	SetColorPalette(color_start, color_end);
	SetAlphaPalette(alpha_start, alpha_end);
}

...

void TransferFunction::SetColorPallete(int color_start[3], int color_end[3])
{
    for(int i=0; i<3; i++)
    {
        if(!m_PalleteC[i])
        {
            m_PalleteC[i] = shared_ptr<float>(new float[MAX_INTENSITY]);
            memeset(m_PalleteC[i],0,sizeof(float)*MAX_INTENSITY);
        }

        for(int j=color_start[i]; j<color_end[i]; j++)
        {
            float start = static_cast<float>(color_start[i]);
            float end = static_cast<float>(color_end[i]);
            m_PalleteC[i].get()[j] = (static_cast<float>(j) - start) / (end - start) * 255.f;
        }

        for(int j=color_end[i]; j<255; j++)
        {
            m_PalleteC[i].get()[j] = 255.f;
        }
    }
}

void TransferFunction::SetAlphaPalette(int alpha_start, int alpha_end)
{
	if (!m_PaletteA)
	{
		m_PaletteA = shared_ptr<float>(new float[MAX_INTENSITY]);
		memset(m_PaletteA.get(), 0, sizeof(float)*MAX_INTENSITY);
	}

	for (int i = alpha_start; i < alpha_end; i++)
	{
		float start = static_cast<float>(alpha_start);
		float end = static_cast<float>(alpha_end);
		m_PaletteA.get()[i] = (static_cast<float>(i) - start) / (end - start);
	}

	for (int i = alpha_end; i < 255; i++)
	{
		m_PaletteA.get()[i] = 1.f;
	}
}

float GetPalleteCValue(int color, int intensity)
{
    if(intensity >= MAX_INTENSITY) return -1.f;
    switch(color)
    {
    case 0:
        return m_PalleteC[0].get()[intensity];
        break;
    case 1:
        return m_PalleteC[1].get()[intensity];
        break;
    case 2:
        return m_PalleteC[2].get()[intensity];
        break;    
    default: /// color 값이 잘못들어오면... return -1.f;
        return -1.f;
        break;
    }
}

float GetPalleteAValue(int intensity)
{
    if(intensity >= MAX_INTENSITY) return -1.f;
    return m_PalleteA.get()[intensity];
}
{% endhighlight %}