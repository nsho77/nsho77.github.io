---
layout: post
title: 여러 방향에서 만드는 MIP 의 속도를 향상시켜보자.
date: 2018-07-24 11:20:00 +0900
description: MIP Any Direction 의 속도 를 향상시켜보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing, volume]
categories: [Image Processing]
---

속도를 향상시키기 전에 현재 얼마나 걸리는 지 측정해보자.
> VolumeRendererDoc.cpp
{% highlight cpp %}
...
/// 다음의 변수를 선언한다.
LARGE_INTEGER Frequency;
LARGE_INTEGER BeginTime;
LARGE_INTEGER Endtime;
...

/// 성능을 측정하고자 하는 부분에 코드를 추가하고
/// 시간을 출력한다.
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

	m_pRenderer->RenderMIPAnyDirection(image.get(), img_width, img_height);
	

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

내 컴퓨터에서 시간은 약 14.04 초 로 나온다.

시간을 줄여보자 시간이 가장 많이 잡아먹는 부분은 함수 RenderMIPAnyDirection에서 3중반복문 안의
view_vector 만큼 진행되면서 1000번 반복하는 부분이다. 이 부분을 적당한 값만큼 반복하는 것으로 만들 수 있다면 걸리는
시간이 줄어들 것이다.

cur_coord = cur_coord + k*view_vector  이라면 k 값을 구하여 반복횟수를 줄일 수 있을 것이다.

1. x,y,z 각각 cur_coord 가 volume에 접근하는 k 값을 구한다. 
2. x,y,z 각각 cur_coord 가 volume에서 나가는 k 값을 구한다.
3. 접근할 때의 k 값은 1의 max 값이다.
4. 나갈 때의 k 값은 2의 min 값이다.

이를 구현해보자.
> Renderer.cpp
{% highlight cpp %}
void Renderer::GetRayBound(float t[2], float3 start_coord, float3 view_vector)
{
	const float EPS = 0.00001f;
	/// 1 2를 구한다.
	float kx[2] = { -20000, 20000 }, ky[2] = { -20000, 20000 }, kz[2] = { -20000, 20000 };

	if (fabs((float)view_vector.x) > EPS)
	{
		kx[0] = (0.f - start_coord.x) / view_vector.x;
		kx[1] = (m_pVolume->GetWidth() - start_coord.x) / view_vector.x;
		if (kx[0] > kx[1])
		{
			/// in > out
			float temp = kx[0];
			kx[0] = kx[1];
			kx[1] = temp;
		}
	}

	if (fabs((float)view_vector.y) > EPS)
	{
		ky[0] = (0.f - start_coord.y) / view_vector.y;
		ky[1] = (m_pVolume->GetHeight() - start_coord.y) / view_vector.y;
		if (ky[0] > ky[1])
		{
			/// in > out
			float temp = ky[0];
			ky[0] = ky[1];
			ky[1] = temp;
		}
	}

	if (fabs((float)view_vector.z) > EPS)
	{
		kz[0] = (0.f - start_coord.z) / view_vector.z;
		kz[1] = (m_pVolume->GetDepth() - start_coord.z) / view_vector.z;
		if (kz[0] > kz[1])
		{
			/// in > out
			float temp = kz[0];
			kz[0] = kz[1];
			kz[1] = temp;
		}
	}

	t[0] = __max(__max(kx[0], ky[0]), kz[0]);
	t[1] = __min(__min(kx[1], ky[1]), kz[1]);

	t[0] = t[0] + 0.01f;
	t[1] = t[1] - 0.01f;

}
{% endhighlight %}

RenderMIPAnyDirection 함수도 수정한다.
{% highlight cpp %}
...
    for (int j = 0; j < img_height; j++)
	{
		///시작 픽셀의 3차원 좌표 계산
		float3 cur_coord = eye_coord + 
			x_vector * (i-img_width/2) + y_vector * (j-img_height/2);

        /// volume 에 만나고 나가는 적절한 k값 범위를 구함    
		float t[2] = { 0.f, 0.f };
		GetRayBound(t, cur_coord, view_vector);
		unsigned char max_value = 0;
		for (float k = t[0]; k < t[1]; k += 1.f)
		{
            /// 현재 좌표를 다시 구함
			float3 adv_coord = cur_coord + view_vector * k;
			///진행 픽셀의 현재 위치가 볼륨 바운더리 안에 들어왔다면 
			if (adv_coord.x >= 0.f && adv_coord.x < vol_width-1 &&
				adv_coord.y >= 0.f && adv_coord.y < vol_height-1 &&
				adv_coord.z >= 0.f && adv_coord.z < vol_depth-1)
			{
				///해당 위치에서의 볼륨 복셀을 가져옴
				float voxel = 
					m_pVolume->GetVoxel(adv_coord.x, adv_coord.y, adv_coord.z);
...

{% endhighlight %}

실행하면 결과는 약 10.55 초 가 나온다. 약 4초 줄어든 것을 확인할 수 있다.