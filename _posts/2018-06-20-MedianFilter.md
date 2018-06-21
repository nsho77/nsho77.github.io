---
layout: post
title: Image에 Median Filter를 적용해보자
date: 2018-06-18 08:30:00 +0900
description: mfc imageprocessing Median Filter 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
---

median filter는 window masking 영역의 값중 중간 값을 해당 픽셀에 넣는 다른 형태의 마스킹이다.
중간값을 구하기 위해 배열 정렬이 필요한데, 정렬 알고리즘 중 가장 빠르다는 radix sort와 kth 번째 큰 값을
찾을 때 빠른 성능을 보여주는 median of Medians 알고리즘 둘 다 이용하고 시간을 측정해보자.

기본 구조는 window 마스킹과 다르지 않다. ksize를 입력받고 그 범위만큼의 값들로 중간값을 구한다.
> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void MedianFilterSingleChannel(unsigned char* image_input,
		const int width, const int height, int ksize);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::MedianFilterSingleChannel(unsigned char* image_input,
	const int width, const int height, int ksize)
{
	if (ksize % 2 == 0 || ksize == 1) return;

	unsigned char* temp = new unsigned char[width*height];
	memcpy(temp, image_input, sizeof(unsigned char)*width*height);

    // 양 옆 neighbor 만큼 윈도우를 만든다.
	int neighbor = ksize / 2;

// 아래는 병렬처리를 위한 지시어 이다. open mp 사용을 체크하여야 사용할 수 있다.
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
            // 중간값을 구할 배열
			vector<unsigned char> medianArr;
			medianArr.resize(ksize*ksize);
			for (int x = -neighbor; x <= neighbor; x++)
			{
				for (int y = -neighbor; y <= neighbor; y++)
				{
                    // 바운더리 처리
					if (x + i >= width || x + i < 0 || y + j >= height
						|| y + j < 0) continue;

                    // 배열에 윈도우에 해당되는 값들을 저장한다.
					medianArr[ksize*(y + neighbor) + (x + neighbor)] =
						image_input[width*(y + j) + (x + i)];
				}
			}

            // 정렬 알고리즘을 선택한다. 현재는 radix sort 이다.

				//radix sort medianArr
				MyRadixSort(medianArr);
				temp[width*j + i] = medianArr[medianArr.size() / 2];


				//median of medians
				//temp[width*j + i] = MedianOfMedians(&medianArr[0], medianArr.size(), medianArr.size() / 2);
				
		}
	}

	memcpy(image_input, temp, sizeof(unsigned char)*width*height);
	delete[] temp;
}
{% endhighlight %}

이제 정렬 알고리즘인 radix sort를 구현하자.
> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void MyRadixSort(vector<unsigned char>& arr);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::MyRadixSort(vector<unsigned char>& arr)
{
    int bucket[256] = {0};
    // bucket 에 빈도를 체크한다.
    for(int i=0; i<arr.size(); i++)
        bucket[arr[i]] ++;

    // arr 을 초기화한다.
    arr.clear();

    // arr 을 정렬한다.
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<bucket[i]; j++)
            arr.push_back(i);
    }    
}
{% endhighlight %}

단일 채널들을 모아서 color image에 마스킹을 적용하는 함수를 만들자.
> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public:
    static void MedianFilter(unsigned char* image_color,
		const int width, const int height, int ksize);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
void ImageProc::MedianFilter(unsigned char* image_color,
		const int width, const int height, int ksize)
{
	unsigned char* img_R = new unsigned char[width*height];
	unsigned char* img_G = new unsigned char[width*height];
	unsigned char* img_B = new unsigned char[width*height];

	SplitChannels_ColorToRGB(img_R, img_G, img_B, image_color, width, height);

	MedianFilterSingleChannel(img_R, width, height, ksize, numOfSortMethod);
	MedianFilterSingleChannel(img_G, width, height, ksize, numOfSortMethod);
	MedianFilterSingleChannel(img_B, width, height, ksize, numOfSortMethod);

	MergeChannels_RGBToColor(img_R, img_G, img_B, image_color, width, height);

	delete[] img_R;
	delete[] img_G;
	delete[] img_B;    
}
{% endhighlight %}



이벤트 처리기를 만들고 시간을 구하는 코드를 작성해보자.
> ImageProcessingDoc.h
{% highlight cpp %}
class CImageProcessingDoc : public CDocument
{
    ...
    afx void afx_msg void OnMedianfilter();
}
{% endhighlight %}

> ImageProcessingDoc.cpp
{% highlight cpp %}
...
LARGE_INTEGER Frequency;
LARGE_INTEGER BeginTime;
LARGE_INTEGER Endtime;
...

void CImageProcessingDoc::OnMedianfilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	printf("MedianFilter RadixSort\n");

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&BeginTime);

	ImageProc::MedianFilter(m_Images[cur_index].image_color,
		m_Images[cur_index].width, m_Images[cur_index].height, 5);

	QueryPerformanceCounter(&Endtime);
	int elapsed = Endtime.QuadPart - BeginTime.QuadPart;
	double duringtime = (double)elapsed / (double)Frequency.QuadPart;

	printf("MedianFilter time : %f\n", duringtime);
	ImageProc::MergeChannels(m_Images[cur_index].image_color,
		m_Images[cur_index].image_gray, m_Images[cur_index].width, m_Images[cur_index].height);

	CImageProcessingView* pView = (CImageProcessingView*)((CMainFrame*)(AfxGetApp()->m_pMainWnd))->GetActiveView();

	pView->SetDrawImage(m_Images[cur_index].image_color, m_Images[cur_index].image_gray,
		m_Images[cur_index].width, m_Images[cur_index].height);

	pView->OnInitialUpdate();
	printf("MedianFilter RadixSort End \n");
}
{% endhighlight %}

실행하면 다음과 같이 이미지가 처리된다.
![medianFilter-hijy]({{"/assets/img/imageProcessing/medianFilter-hijy.jpg"}})

콘솔창을 보면 
MedianFilter time : 15.339769
MedianFilter MedianOfMedians End
으로 약 15초 걸린다.

median of medians 알고리즘은 다음과 같다.
median of medians 에 대한 자세한 이해는 [다음의 포스터](../Median-of-medians)를 참고하자.

> ImageProc.h
{% highlight cpp %}
class ImageProc
{
    ...
public :
    static int MedianOfMedians(unsigned char *v, int n, int k);
}
{% endhighlight %}

> ImageProc.cpp
{% highlight cpp %}
int ImageProc::MedianOfMedians(unsigned char *v, int n, int k) {
	if (n == 1 && k == 0) return v[0];

	int m = (n + 4) / 5;
	unsigned char *medians = new unsigned char[m];
	for (int i = 0; i<m; i++) {
		if (5 * i + 4 < n) {
			unsigned char *w = v + 5 * i;
			for (int j0 = 0; j0<3; j0++) {
				int jmin = j0;
				for (int j = j0 + 1; j<5; j++) {
					if (w[j] < w[jmin]) jmin = j;
				}
				mySwap(w[j0], w[jmin]);
			}
			medians[i] = w[2];
		}
		else {
			medians[i] = v[5 * i];
		}
	}

	int pivot = MedianOfMedians(medians, m, m / 2);
	delete[] medians;

	for (int i = 0; i<n; i++) {
		if (v[i] == pivot) {
			mySwap(v[i], v[n - 1]);
			break;
		}
	}

	int store = 0;
	for (int i = 0; i<n - 1; i++) {
		if (v[i] < pivot) {
			mySwap(v[i], v[store++]);
		}
	}
	mySwap(v[store], v[n - 1]);

	if (store == k) {
		return pivot;
	}
	else if (store > k) {
		return MedianOfMedians(v, store, k);
	}
	else {
		return MedianOfMedians(v + store + 1, n - store - 1, k - store - 1);
	}
}
{% endhighlight %}

radix sort 호출 부분에 주석처리 하고 median of medians 함수를 호출하면 적용된다.

실행하여 콘솔창을 보면 
MedianFilter time : 8.011197
MedianFilter MedianOfMedians End
으로 약 8초 걸린다.

Median of Medians 알고리즘 적용한것이 radix sort 보다 성능이 좋다는 것을 알 수 있다.