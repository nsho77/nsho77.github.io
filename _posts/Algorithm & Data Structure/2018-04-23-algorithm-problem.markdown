---
layout: post
title: 알고리즘 3개 풀이
date: 2018-04-23 14:45:00 +0900
description: 백준에서 규칙찾기 문제 3개를 풀어보았다. 마크다운 도 연습할겸.. # Add post description (optional)
img: software.jpg # Add image post (optional)
tags: [Algorithm, Code] # add tag
categories : [Algorithm & Data Structure]
---

코딩 연습 할 겸 백준(<https://www.acmicpc.net/>)에서 문제를 찾아 풀기 시작했다. 목표는 일주일에 3개!! Slack과 블로그에 코딩한 것과 코멘트를 올릴 생각이다.

### 문제 1

첫째 줄에는 별 1개, 둘째 줄에는 별 2개, N 번째 줄에는 N개의 별을 찍는 문제

```cpp
#include <stdio.h>

/// 매개변수 만큼의 라인의 별을 찍는다
void printStar(int n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<i+1; j++)
        {
            /// n번째 라인에 n개의 별을 찍는다
            printf("*");
        }
        printf("\n");
    }
}

int main()
{
    int n = 0;
    /// 키보드 입력을 받는다.
    scanf("%d", &n);
    printStar(n);
}
```


### 문제 2

<div style="overflow:hidden;">
<img style="float:left; width:30%" src="https://www.acmicpc.net/JudgeOnline/upload/201009/3(2).png" alt="벌집문제"/>
<div>
왼쪽 그림과 같이 육각형으로 이루어진 벌집이 있다. 그림에서 보는 바와 같이 중앙의 방 1부터 시작해서 이웃하는 방에 돌아가면서 1씩 증가하는 번호를 주소로 매길 수 있다. 숫자 N이 주어졌을 때, 벌집의 중앙 1에서 N번 방까지 최소 개수의 방을 지나서 갈 때 몇 개의 방을 지나가는지(시작과 끝을 포함하여)를 계산하는 프로그램을 작성하시오. 예를 들면, 13까지는 3개, 58까지는 5개를 지난다.
</div>
</div>
```cpp
#include <stdio.h>

/// 거리가 1 만큼 멀어질 때마다 거리*6 의 벌집이 생긴다
/// n번째 거리의 벌집수 = 1 + 1*6 + 2*6 + 3*6 + ... + (n-1)*6
void printNumberOfRoom(int n)
{
	/// 거리에서 가장 높은 인덱스를 저장(인덱스 == 방 number)
	int maxNum = 1;
	/// 거리 = resCnt
	int resCnt = 1;

	/// 가장 높은 인덱스가 
	/// 찾고자 하는 인덱스 보다 크면 반복문 종료
	while (maxNum < n)
	{
		maxNum += 6*resCnt;
		resCnt++;
	}

	printf("%d",resCnt);
}

int main()
{
	int n = 0;
	scanf("%d", &n);
	printNumberOfRoom(n);
    return 0;
}
```

### 문제 3
<table style="width:50%">
    <tr>
    <td>1/1</td><td>1/2</td><td>1/3</td><td>1/4</td><td>1/5</td><td>...</td>
    </tr>
    <tr>
    <td>2/1</td><td>2/2</td><td>2/3</td><td>2/4</td><td>2/5</td><td>...</td>
    </tr>
    <tr>
    <td>3/1</td><td>3/2</td><td>3/3</td><td>3/4</td><td>3/5</td><td>...</td>
    </tr>
    <tr>
    <td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>
    </tr>
</table>
이와 같이 나열된 분수들을 1/1 -> 1/2 -> 2/1 -> 3/1 -> 2/2 -> … 과 같은 순서로 차례대로 1번, 2번, 3번, 4번, 5번, … 분수라고 하자.

X가 주어졌을 때, X번째 분수를 구하는 프로그램을 작성하시오.

```cpp
#include <stdio.h>
/// 행렬을 대각선 부터 지그재그로 돌면서
/// 분수를 프린트하는 함수
/// n 번째 라인에는 n개의 요소가 있음.
void printBunsu(int n)
{
	/// 순회한 모든 요소의 갯수.
	int indexCnt = 0;
	/// 몇 번째 라인인지 저장.
	int incrementNum = 0;
	
	/// 순회한 요소가 찾으려는 인덱스보다 많으면
	/// 반복문 종료.
	while (indexCnt < n)
	{
		incrementNum++;
		indexCnt += incrementNum;
	}

	/// indexCnt는 라인의 끝을 가리키고 있음.
	/// 찾으려는 요소와 얼마나 떨어져 있는지 저장.
	int moveCnt = indexCnt - n;

	/// 요소가 있는 라인이 홀수인지 짝수인지에 따라
	/// 분모와 분자를 반대로 설정.
	int mo = incrementNum;
	int ja = 1;

	if (incrementNum % 2 == 0)
	{
		mo = 1;
		ja = incrementNum;
	}

	/// 짝수번째 이면 분모에 찾으려는 인덱스와 떨어진 만큼
	/// 더하고 분자는 뺀다.
	/// 홀수번째 이면 반대.
	if (incrementNum % 2 == 0)
	{
		mo += moveCnt;
		ja -= moveCnt;
	}
	else
	{
		mo -= moveCnt;
		ja += moveCnt;
	}
	
	printf("%d/%d", ja, mo);

}

int main()
{
	int n = 0;
	scanf("%d", &n);
	printBunsu(n);
    return 0;
}
```