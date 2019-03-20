---
layout: post
title: ImageProcessing 기술을 이용하여 적혈구 숫자를 세보자
date: 2018-06-28 14:45:00 +0900
description: mfc imageprocessing region growing 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
categories: [Image Processing]
---

적혈구 사진을 준비하고 image에 나오는 적혈구 숫자를 세어보는 놀이를 해보자.

적혈구 사진
![redbloodcell]({{"/assets/img/imageProcessing/redbloodcell.jpg"}})

작업 순서는 다음과 같다.
1. 이진화를 통해서 적혈구 만 남긴다.
2. 이진화 이미지를 탐색한다.
    2-1. 적혈구에 도달했을 때 region growing 으로 적혈구를 탐색한다. 
    2-2. 최소 좌표값과 최대 좌표값을 체크한다.(크기를 측정하기 위해서)
    2-3. 적혈구 사이즈 인 것만 카운팅한다.
    2-4. 적혈구 가운데에 하얀색 점을 찍는다.

먼저 이미 만들어놓은 이진화 함수와 이진이미지 침식 함수를 이용하여 적혈구만을 남겨보자.
헤더 파일에 선언부는 생략하겠다.

1. 이진화를 통해서 적혈구만 남긴다.

> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnPracticeCountredbloodcell()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
    // 245로 이진화를 한다.
	ImageProc::Binarization(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height,245);
    // window 크기 3으로 침식처리한다.
	ImageProc::BinaryErosion(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 3);
    
	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 1);

	pView->OnInitialUpdate();
}
{% endhighlight %}

실행하면 다음과 같은 이미지가 만들어진다.
![binary_erosion_cell]({{"/assets/img/imageProcessing/binary_erosion_cell.jpg"}})

이진화 이미지를 탐색해서 적혈구의 숫자를 세보자.
> ImageProc.cpp
{% highlight cpp %}
void ImageProc::CountRedBloodCell(unsigned char* image_input,
	const int width, const int height)
{
	// 좌표값을 저장할 수 있는 구조체를 정의한다.
	struct POINT
	{
		int x;
		int y;
		POINT(void)
		{
			x = -1;
			y = -1;
		}
		POINT(int _x, int _y)
		{
			x = _x;
			y = _y;
		}
	};
	// 적혈구 숫자 카운트하는 변수
	int cnt = 0;
	// 적혈구를 찾을 수 없을 때까지 반복한다.
	while (1)
	{
		// 0 인 픽셀에 도달하면
		// 해당 좌표를 시작지점으로
		// region growing 을 시작한다.

		// seed 선언
		POINT seed = POINT();

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				// 픽셀 값이 0 이면
				// seed에 좌표 저장
				if (image_input[j*width + i] == 0)
				{
					seed = POINT(i, j);
					break;
				}
			}
			// seed 값을 찾으면 seed 찾는 반복문 종료
			// 못찾으면 끝까지 탐색
			if (seed.x != -1 && seed.y != -1) break;
		}
		
		// 끝까지 탐색해도 못찾으면
		// 이진 이미지 탐색 종료
		if (seed.x == -1 && seed.y == -1) break;
		
		// seed 를 바탕으로 region growing 을 시작함
		// region growing 이란 그래프 탐색에서 너비우선 탐색을 의미함.
		// 탐색하면서 좌표의 최소값과 최대값을 저장한다.
		// 이미 탐색한 적혈구는 gray 색으로 체크한다.
		vector<POINT> queue;
		queue.push_back(seed);

		// 좌표의 최대값과 최소값을 저장할 변수
		POINT minPoint = POINT(width-1, height-1);
		POINT maxPoint = POINT(0, 0);

		// queue 가 완전히 비워질때까지 탐색한다.
		while (!queue.empty())
		{
			POINT cur_seed = POINT(queue[0].x, queue[0].y);

			// 주위 1pixel 을 보며 queue에 넣고 체크한다.
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					if((cur_seed.y + y)>= height || (cur_seed.y + y) <0
						|| (cur_seed.x + x) >= width || (cur_seed.x + x) < 0) continue;

					if (image_input[width*(cur_seed.y + y)
						+ (cur_seed.x + x)] == 0)
					{
						// 방문체크
						image_input[width*(cur_seed.y + y)
							+ (cur_seed.x + x)] = 127;

						// 최소, 최대값 체크
						minPoint.x = __min(minPoint.x, cur_seed.x + x);
						minPoint.y = __min(minPoint.y, cur_seed.y + y);
						maxPoint.x = __max(maxPoint.x, cur_seed.x + x);
						maxPoint.y = __max(maxPoint.y, cur_seed.y + y);

						// queue push back
						queue.push_back(POINT(cur_seed.x + x, cur_seed.y + y));
					}
				}
			}
			
			// queue의 가장 앞부분을 pop 한다.
			queue.erase(queue.begin());
		}

		// 최대 최소값을 통해 적혈구 여부를 파악하고 카운팅한다.
		// 적혈구면 가운데 하얀점을 찍는다.
		float cell_width = maxPoint.x - minPoint.x;
		float cell_height = maxPoint.y - minPoint.y;
		if ( cell_height == 0
			|| cell_width / cell_height > 2 
			|| cell_width / cell_height < 0.5 ) continue;
		// 적혈구 숫자 카운팅
		cnt++;
		// 가운데 점 찍기
		POINT center = POINT((maxPoint.x + minPoint.x) / 2,
			(maxPoint.y + minPoint.y) / 2);
		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				image_input[(center.y+y)*width + center.x+x] = 255;
			}
		}
		
	}

	printf("red blood cell cnt : %d\n", cnt);
}
{% endhighlight %}

이벤트 처리기를 수정해보자
> ImageProcessingDoc.cpp
{% highlight cpp %}
void CImageProcessingDoc::OnPracticeCountredbloodcell()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

    // 침식할 경우 적혈구 사이의 거리가 너무 가까워 침식 적용 하지 않았다.
    // 이진화 처리에 threshold 값을 약간 조절 했다.
	ImageProc::Binarization(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height,185);


	/*ImageProc::BinaryErosion(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 3);*/


	ImageProc::CountRedBloodCell(m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height, 1);

	pView->OnInitialUpdate();
}
{% endhighlight %}

두 적혈구의 거리가 너무 가깝거나 붙어있는 경우 두 적혈구 사이에 하얀점이 찍히는 문제가 있었다.
아래는 위 코드를 실행한 결과이다.

![count_cell]({{"/assets/img/imageProcessing/count_cell.jpg"}})

갯수는
red blood cell cnt : 252
라고 나왔다.