---
layout: post
title: fly me to the alpha... 문제풀이
date: 2018-04-29 12:00:00 +0900
description: 백준의 문제를 풀어보았다. # Add post description (optional)
img:  we-in-rest.jpg # Add image post (optional)
tags: [Algorithm, Code] # add tag
---
백준에 있는 fly_me_to_the_alpha_centauri 를 풀어보았다. <br />
사실 풀지 못하고 사람들이 풀이해준 걸 보고 풀었다. 그래도 어려웠다... <br />
[문제링크](https://www.acmicpc.net/problem/1011)
```cpp
#include <stdio.h>
#include <math.h>

/// x부터 y지점까지 도달하기 까지의 최소 워프 수를 출력하는 문제
/// 거리 차가 제곱 수 일 때 최대 워프 수가 1씩 늘어나는 규칙이 있음.
/// 거리차  최소워프수				최대 워프 거리
/// ex) 1 -> 1 ... 1						1
///		2 -> 2 ... 1 1						1
///		3 -> 3 ... 1 1 1					1
///		4 -> 3 ... 1 2 1					2
///		9 -> 5 ... 1 2 3 2 1				3
/// 거리차가 n^2 이면 최대 워프 거리는 n임.
/// 거리차가 n^2 이면 최소워프 수는 2*n -1 임.
///	최대 워프 이상 움직일 수 없음.
/// 나머지 거리를 구할 때는 나머리 거리를 최대 워프 거리로 나눈 몫을 올림하면 됌
/// ex) 거리가 5일때 -> 최대워프 거리가 2 이므로 2^2 거리를 가고 
/// 나머지 1거리는 최대 워프 거리로 나눈 몫을 올림한 값. 따라서 + 1 ... 3
/// 1 2 1 1
void fly_me_to_the_alpha_centauri(int x, int y)
{
	int goToPath = y - x;
	int standard = 1; /// 제곱이 될 수
	int maxWarp;
	int warpCnt;

	while (goToPath > pow(standard,2)) /// 남은 거리보다 크지 않은 n^2 의 n을 구한다.
		standard++;

	maxWarp = --standard; /// 구해야 할 n보다 1크므로 감소시켜 준다. 최대 워프 거리와 같다.
	goToPath -= pow(standard, 2); /// 앞으로 이동해야 할 거리를 구한다.
	warpCnt = 2 * standard - 1; /// 지금까지 이동한 워프 수를 구한다.

	while (goToPath > 0)	/// maxWarp로 나눈 몫을 올림한다. 남은 거리를 maxWarp로 빼면서 이동 거리를 1씩
	                        /// 증가 시킨다.
	{
		goToPath -= maxWarp;
		warpCnt++;
	}
		
	printf("%d", warpCnt);
}

int main()
{
	int cnt;
	scanf_s("%d", &cnt);
	for (int i = 0; i < cnt; i++)
	{
		int x;
		int y;
		scanf_s("%d", &x);
		scanf_s("%d", &y);
		fly_me_to_the_alpha_centauri(x, y);
		printf("\n");
	}
    return 0;
}
```
