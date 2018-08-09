---
layout: post
title: 4가지 통계값 찾기 문제를 다른방법으로 풀어보자
date: 2018-08-02 11:00:00 +0900
description: 내가 구현한 4가지 통계값 찾기 # Add post description (optional)
img:  # Add image post (optional)
tags: [Algorithm, Programming]
---

이전 포스팅에서 풀어보았던 4가지 통계값 찾기 문제를 다른 방법으로 풀어보자

첫번째 바꿔보고 싶은 것은 C언어에서 반올림 방법을 익힌 뒤 산술평균 문제에 적용하는 것이다.

반올림 대상 값으로 음수가 들어올 수 있음을 유의하자.

## C언어에서 반올림 하는 방법

출력만 하는 경우는 아래와 같이 형식지정자를 쓰면 반올림이 된다.

{% highlight cpp %}
float myFloat = 37.7777779f;
printf("%.2f", myFloat);
/// 출력 37.78
{% endhighlight %}

연산 등을 위해 반올림하는 경우 math.h를 사용한다.
{% highlight cpp %}
#include <math.h>
float val = 37.777779f;
float rounded_down = floatf(val*100) / 100; /// 내림 37.77
float nearest = roundf(val*100) / 100; /// 반올림 37.78
float rounded_up = ceilf(val*100) / 100; /// 올림 37.78
{% endhighlight %}

몇 번째 자리에서 반올림 할 것인지에 따라 나눠주고 곱해주면 된다.

함수를 사용하는 것보다 사칙연산을 사용하는 것이 시간이 적게 걸리므로 이 부분은 수정 하지 않는 게 좋다고 결론냈다. 음수인 경우 양수와 반대로 -.5f 를 더해주는 것을 잊지말자.

두 번째로 고치고 싶은 것은 입력받은 숫자를 배열에 저장한 뒤 이 배열을 오름차순으로 정렬해서 중앙값을 찾는 것이다.

이를 위해 이용할 알고리즘은
1. Merge Sort
2. Heap Sort
3. Quick Sort


모두 정렬하지 않고 k 번째 값을 찾는 
1. Quick Select
2. Medium of Medium

우선 Merge Sort 를 구현하여 중앙값을 찾아보자.
{% highlight cpp %}
#include <stdio.h>
#include <string.h>

int FindMiddleValue(int* arr, int size);
void Merge(int* arr, int left, int middle, int right);

int main()
{
	int cnt; scanf("%d", &cnt);
	int inputNum;

	int res_average = 0;
	float average_f = 0.f;
	int mediumIndx = cnt / 2;
	int bucket[8001] = { 0 };
	int max_value = -4001;
	int min_value = 40001;
	int range = 0;

	int* forMiddleArray = new int[cnt];

	for (int i = 0; i < cnt; i++)
	{
		scanf("%d", &inputNum);
		
		average_f += inputNum;

		bucket[inputNum + 4000]++;
		
		if (inputNum < min_value)
			min_value = inputNum;
		if (inputNum > max_value)
			max_value = inputNum;

		forMiddleArray[i] = inputNum;
	}

	/// 산술평균
	if (cnt != 0)
	{
		if (average_f < 0)
			res_average = average_f / cnt - 0.5f;
		else
			res_average = average_f / cnt + 0.5f;
	}
	printf("%d\n", res_average);

	/// 중앙값
	int middleValue = FindMiddleValue(forMiddleArray,cnt);
	printf("%d\n", middleValue);

	/// 최빈값
	int manyValue = -4001;
	int maxCnt = 0; int isEqual = 0;
	for (int i = 0; i < 8001; i++)
	{
		if (bucket[i] > maxCnt)
		{
			manyValue = i;
			maxCnt = bucket[i];
			isEqual = 0;
		}
		else if (bucket[i] == maxCnt)
			isEqual = 1;
	}
	int second = 0;
	if (isEqual)
	{
		for (int i = 0; i < 8001; i++)
		{
			if (bucket[i] == maxCnt)
				second++;
			if (second == 2)
			{
				printf("%d\n", i - 4000);
				break;
			}
		}
	}
	else if (isEqual == 0)
		printf("%d\n", manyValue - 4000);

	/// 범위
	range = max_value - min_value;
	printf("%d\n",range );

	delete[] forMiddleArray;
	return 0;
}


void MergeSort(int* arr, int left, int right)
{
	if (left >= right) return;

	int middle = (left + right) / 2;

	MergeSort(arr, left, middle);
	MergeSort(arr, middle + 1, right);

	Merge(arr, left, middle, right);
}

void Merge(int* arr, int left, int middle, int right)
{
	int sortedSize = right - left + 1;
	int* sortedArray = new int[sortedSize];
	int leftIndex = left, rightIndex = middle+1;
	int sortedIndex = 0;
	while (leftIndex <= middle && rightIndex <= right)
	{
		if (arr[leftIndex] < arr[rightIndex])
		{
			sortedArray[sortedIndex] = arr[leftIndex];
			leftIndex++;
		}
		else
		{
			sortedArray[sortedIndex] = arr[rightIndex];
			rightIndex++;
		}
		sortedIndex++;
	}
	while (leftIndex <= middle)
	{
		sortedArray[sortedIndex++] = arr[leftIndex++];
	}
	while (rightIndex <= right)
	{
		sortedArray[sortedIndex++] = arr[rightIndex++];
	}

	memcpy(&arr[left], sortedArray, sizeof(int)*sortedSize);

	delete[] sortedArray;
}


int FindMiddleValue(int* arr, int size)
{
	MergeSort(arr,0, size-1);
	return arr[size / 2];
}

{% endhighlight %}

속도가 더 느려지지만 MergeSort로도 충분히 해결할 수 있다.

다음은 Quick Select 를 구현해 활용해보자.
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