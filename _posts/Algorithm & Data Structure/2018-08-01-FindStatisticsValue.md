---
layout: post
title: 4가지 통계값 찾기 문제를 풀어보자
date: 2018-08-01 15:30:00 +0900
description: 내가 구현한 4가지 통계값 찾기 # Add post description (optional)
img:  # Add image post (optional)
tags: [Algorithm, Programming]
---

백준의 2108번 문제를 풀어보았다.

문제는
1. 산술평균 구하기
2. 중앙값 구하기
3. 최빈값 구하기
4. 범위

이다.
자세한 문제 사항은 이 [링크](https://www.acmicpc.net/problem/2108)를 참조하자

내가 만든 소스부터 블로깅하고

다른 사람이 만든 소스는 분석하여 다시 블로깅 하겠다.

{% highlight cpp %}
int main()
{
	int cnt; scanf_s("%d", &cnt);
	int inputNum;

	int res_average = 0;
	float average_f = 0.f;
	int mediumIndx = cnt / 2;
	int bucket[8001] = { 0 };
	int max_value = -4001;
	int min_value = 40001;
	int range = 0;

	for (int i = 0; i < cnt; i++)
	{
		scanf_s("%d", &inputNum);

		average_f += inputNum;

		bucket[inputNum + 4000]++;
		
		if (inputNum < min_value)
			min_value = inputNum;
		if (inputNum > max_value)
			max_value = inputNum;
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
	int mediumCnt = 0;
	int mediumValue = 0;
	int mediumChecked = 0;

	/// 최빈값
	int manyValue = -4001;
	int maxCnt = 0; int isEqual = 0;
	for (int i = 0; i < 8001; i++)
	{
		if (!mediumChecked)
		{
			mediumCnt += bucket[i];
			if (mediumCnt - 1 >= mediumIndx)
			{
				mediumValue = i;
				mediumChecked = 1;
			}
		}

		if (bucket[i] > maxCnt)
		{
			manyValue = i;
			maxCnt = bucket[i];
			isEqual = 0;
		}
		else if (bucket[i] == maxCnt)
			isEqual = 1;
	}

	printf("%d\n", mediumValue-4000);

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

	return 0;
}
{% endhighlight %}