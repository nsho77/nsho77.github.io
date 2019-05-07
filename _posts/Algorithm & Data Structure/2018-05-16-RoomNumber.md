---
layout: post
title: 방번호 문제 풀기
date: 2018-05-16 14:45:00 +0900
description: 백준문제 풀기 # Add post description (optional)
img:  # Add image post (optional)
tags: [Algorithm, Code]
categories : [Algorithm & Data Structure]
---

> 백준에 있는 방번호 풀어보았다. [문제링크](https://www.acmicpc.net/problem/1475)

크게 두 가지 방법이 있는거 같다.

먼저 string으로 받아서 쪼개는 방법을 해보았다.

{% highlight cpp %}
void roomNumber()
{
	char roomNumber_[8];
	scanf_s("%s",roomNumber_,8);

	const int length = strlen(roomNumber_);

	//하나하나 쪼개기
	int* roomNumber = new int[length];

	for (int i = 0; i < length; i++)
	{
		char a = roomNumber_[i];
		roomNumber[i] = atoi(&a);
		printf("%d\n",roomNumber[i]);
	}

	int countArray[10] = { 0 };

	// 숫자 세기
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < length; j++)
		{
			if (i == roomNumber[j])
				countArray[i]++;
		}
	}

	// 6과 9처리
	int sixAndNine = countArray[6] + countArray[9];
	int sixAndNineNamo = sixAndNine % 2;
	int sixAndNineMok = sixAndNine / 2;
	
	if (sixAndNineNamo != 0)
		sixAndNine = sixAndNineMok + 1;
	else
		sixAndNine = sixAndNineMok;

	countArray[6] = sixAndNine;
	countArray[9] = sixAndNine;

	// 맥스 값 찾기
	int max = 0;
	for(int i=0; i<10; i++)
	{
		if(max < countArray[i])
		{ 
			max = countArray[i];
		}
	}

	printf("%d", max);
	delete[] roomNumber;
}
{% endhighlight %}

string을 어떻게 잘 다룰 건지는 더 연구해보아야 할 것 같다.

여기서 더 똑똑한 방법은 나머지를 이용하는 방법이다. 숫자로 입력을 받은 뒤 10으로 나누어가며 슬라이싱을 하면 된다. 나도 질문게시판을 통해 알았다. 소스코드 공유해주신 분 감사합니다~!

{% highlight cpp %}
void roomNumberUsingNamo()
{
	int cntArray[10] = { 0 };

	int roomNumber = 0;
	scanf_s("%d",&roomNumber);

	int mok = roomNumber / 10;
	int namo = roomNumber % 10;
	
	while (mok != 0 || namo != 0)
	{
		cntArray[namo]++;

		namo = mok % 10;
		mok = mok / 10;
	}

	// 6과 9 처리
	int sixAndNine = cntArray[6] + cntArray[9];
	int sixAndNineNamo = sixAndNine % 2;
	int sixAndNineMok = sixAndNine / 2;

	if (sixAndNineNamo != 0)
		sixAndNine = sixAndNineMok + 1;
	else
		sixAndNine = sixAndNineMok;

	cntArray[6] = sixAndNine;
	cntArray[9] = sixAndNine;

	// max 찾기
	int max = 1;
	for (int i = 0; i < 10; i++)
	{
		if (max < cntArray[i])
			max = cntArray[i];
	}

	printf("%d", max);
}
{% endhighlight %}