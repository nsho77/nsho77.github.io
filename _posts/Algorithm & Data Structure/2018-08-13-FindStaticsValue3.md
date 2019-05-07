---
layout: post
title: 4가지 통계값 찾기 문제를 다른방법으로 풀어보자2 
date: 2018-08-13 10:00:00 +0900
description: 내가 구현한 4가지 통계값 찾기 # Add post description (optional)
img:  # Add image post (optional)
tags: [Algorithm, Programming]
categories : [Algorithm & Data Structure]
---

이전 포스팅에서 Median 값을 찾기 위해 QuickSelect 를 이용했었다. QuickSelect의 경우 pivot 값이 무엇인지에 따라
속도가 달라질 수 있다. 중위값 찾는 함수를  median of median 알고리즘을 이용하여 pivot 값을 찾고 median 값을 구하는 함수로 수정해보자.

다음은 이전에 구현한 중위값을 찾는 함수이다. 여기서는 pivotIndex 를 가장 마지막 인덱스로 지정했다. 이부분을 바꿔보자.
{% highlight cpp %}

int partition(int* arr, int left, int right, int pivotIndex)
{
	swap(arr + pivotIndex, arr + right);
	int res_pivotIdx = right; int l = left;
	for(int i = left; i< right; i++)
	{
		if(arr[i] > arr[res_pivotIdx])
			continue;
		swap(arr+i, arr+l++);
	}

	swap(arr+l , arr+res_pivotIdx);
	res_pivotIdx = l;
	return res_pivotIdx;
}

int QuickSelect(int* arr, int left, int right, int k)
{
	int pivotIndex = right;
	pivotIndex = partition(arr, left, right, pivotIndex);

	if(k < pivotIndex)
		return QuickSelect(arr, left, pivotIndex-1, k);
	else if(k < pivotIndex +1 )
		return arr[pivotIndex];
	else
		return QuickSelect(arr, pivotIndex + 1, right, k);
}

int FindMiddleValue(int* arr, int size)
{
	return QuickSelect(arr, 0, size-1, size /2 );
}
{% endhighlight%}

{% highlight cpp %}


int QuickSelect(int* arr, int left, int right, int k)
{
    /// 이 부분을 수정한다.
	/// int pivotIndex = right;
    int pivot = MedianOfMedian(arr, size);
    int pivotIndex;
    for(int i=0; i<size; i++)
    {
        if(pivot == arr[i])
        {
            pivotIndex = i;
            break;
        }
    }
	pivotIndex = partition(arr, left, right, pivotIndex);

	if(k < pivotIndex)
		return QuickSelect(arr, left, pivotIndex-1, k);
	else if(k < pivotIndex +1 )
		return arr[pivotIndex];
	else
		return QuickSelect(arr, pivotIndex + 1, right, k);
}

{% endhighlight %}

MedianOfMedian 을 구현해보자
{% highlight cpp %}
int MedianOfMedian(int* arr, int size)
{
    if(size < 5)
    {
        for(int i=0; i<size-1; i++)
        {
            int minIndex = i;
            for(int j=1; j<size; j++)
            {
                if(arr[minIndex] > arr[j])
                    minIndex = j;
            }
            swap(arr + i, arr + minIndex);
        }
        return arr[size/2];
    }

    int m = size / 5;
    int* medians = new int[m];
    for(int i=0; i<m; i++)
    {
        int* w = arr + i*5;
        for(int j=0; j<3; j++)
        {
            int minIndex = j;
            for(int k=j+1; k<5; k++)
            {
                if(w[minIndex] > w[k])
                    minIndex = k;
            }
            swap(w + j, w + minIndex);
        }
        medians[i] = w[2];
    }

    int pivot = MedianOfMedian(medians, m);
    delete[] medians;
    return pivot;
}
{% endhighlight cpp %}