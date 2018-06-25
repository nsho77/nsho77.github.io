---
layout: post
title: median of medians 알고리즘을 이해해보자
date: 2018-06-11 09:00:00 +0900
description: median of medians 알고리즘을 이해해보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, algorithm]
---

스택오버플로우 해당 [링크](https://stackoverflow.com/questions/9489061/understanding-median-of-medians-algorithm)에서 median of medians 설명이 있다. 글을 보면서 설명을 이해하고 영상처리에 적용하는게 이번 포스트의 목표이다.

먼저 pseudocode 를 봐보자
{% highlight cpp %}
// L 은 median을 찾아야 할 배열이다.
// k 는 예상 median position 이다. 
// 첫번째 select call은 다음과 같다. select(array, N/2). array는 길이가 N인 배열이다.

select(L, k)
{
    if( L 의 길이가 5이거나 5보다 작으면)
    {
        sort L
        return k 번째 위치의 값
    }

    L 을 5개 원소의 하위 집합으로 나눈다.( n/5 개의 하위집합이 생성).

    for(i = 1 to n/5) do
        x[i] = select(S[i],3)

    M = select({x[i]}, n/10)

    // 이어지는 코드는 M 이 가장 작거나 가장 큰 값이라도 
    // k번째 값을 찾을 수 있도록 해준다.

    partition L into L1 < M, L2 = M, L3 > M

    // 예상 median position k 와 첫번째 배열 L1의 길이를 비교한다.
    // k가 L1 길이보다 작으면 L1 에 대해 select를 재귀호출한다.

    if( k <= length(L1))
        return select(L1, k)

    // k가 L3 배열에 있는지 확인하고 알맞게 재귀호출한다.
    else if( k> length(L1) + length(L2))
        return select(L3, k-length(L1)-length(L2))

    // k가 L2에 있으니 M을 리턴한다.
    else
        return M
}
{% endhighlight %}

다음과 같은 배열이 있다고 해보자
{48 49 50 51 52 43 44 45 46 47 38 39 40 41 42 33 34 35 36 37 28 29 30 31 32 23 24 25 26 27 18 19 20 21 22 13 14 15 16 17 8 9 10 53 54}

median of medians 함수는 45개의 요소가 있는 이 배열에 대해서 다음과 호출 할 것이다.

{% highlight cpp %}
median = select(Array, 45/2)
{% endhighlight %}

1. M = select({x[i]}, n /10) 를 처음 호출할 때, {x[i]}는 다음의 숫자를 가지고 있을 것이다. 50 45 40 35 30 20 15 10
이번 함수 콜에서 n =45 이기 때문에 selection 함수 콜은 M = select({50,45,40,35,30,20,15,10},4) 이 된다.

2. M = select({x[i]}, n /10) 를 두 번째 호출할 때, {x[i]}는 다음의 숫자를 가지고 있을 것이다. 40 20
이번 함수 콜에서 n =9 이기 때문에 selection 함수 콜은 M = select({40, 20},0) 이 된다.

이제 우리는 L 을 M=20과 k=4 를 가지고 partition 할 것이다.

현재 L 은 50 45 40 35 30 20 15 10 이다.

이 배열은 L1, L2, L3 로 L1 < M, L2 = M, L3 > M 기준으로 나뉜다. 따라서
L1 : 10 15
L2 : 20
L3 : 30 35 40 45 50

k =4 이고 length(L1) + length(L2) = 3 보다 크다. 따라서 재귀적으로 다음이 호출된다.
return select(L3, k-length(L1) - length(L2))

는 return select({30, 35, 40, 45, 50}, 1)
과 같고 30 을 return 한다.

이제 전체 배열을 대상으로 select 콜한 값으로 30 return 되고
30으로 배열을 나누는 작업을 거치고 마침내 median of medians를 얻는다.

위의 이해를 바탕으로 아래의 코드를 분석 해보자
{% highlight cpp %}
// k-th 번째 값을 찾으려는 배열 arr 과 arr 의 size, k 값을 받는다.
int ImageProc::MedianOfMedians(unsigned char *v, int n, int k) {

    // 배열에 1 개의 값 밖에 없거나 찾으려는 값이 첫번째 값이면
    // 첫번째 값을 반환한다.
	if (n == 1 && k == 0) return v[0];

    // 배열을 원소 5개의 서브 배열로 나눈다.
    // 올림 하기 위해 배열 사이즈에 4를 더하고 5로 나눈다.
    // m 은 서브 배열 몇 개를 만들어야 하는지 나타낸다.
	int m = (n + 4) / 5;
    // 배열의 주소값을 저장하는 배열 m size 만큼 만든다. 
    // (예를들어 배열 사이즈가 11개라면 ) medians 의 size는 3이다.
    // medians 는 서브배열의 중간값을 저장할 예정.
	unsigned char *medians = new unsigned char[m];
    // 서브 배열의 갯수 만큼 반복한다.
	for (int i = 0; i<m; i++) {
        // 서브 배열중 마지막 서브 배열인지 확인
		if (5 * i + 4 < n) {
            // w 는 서브 배열의 첫번째 값을 가리킴
			unsigned char *w = v + 5 * i;
            // 서브 배열의 반까지만 정렬
			for (int j0 = 0; j0<3; j0++) {
				int jmin = j0;
				for (int j = j0 + 1; j<5; j++) {
					if (w[j] < w[jmin]) jmin = j;
				}
				mySwap(w[j0], w[jmin]);
			}
            // 중간값을 구한 뒤 저장
			medians[i] = w[2];
		}
        // 마지막 서브 배열이면 첫번째 값을 저장
		else {
			medians[i] = v[5 * i];
		}
	}
    // 배열의 중간값들을 이용해 pivot 값을 구한다.
	int pivot = MedianOfMedians(medians, m, m / 2);

    // 필요없는 중간값 배열은 삭제한다.
	delete[] medians;

    //pivot 값을 맨 뒤로 보낸다.
	for (int i = 0; i<n; i++) {
		if (v[i] == pivot) {
			mySwap(v[i], v[n - 1]);
			break;
		}
	}

    // pivot 값과 비교해서 작은 값은 앞에 정렬
	int store = 0;
	for (int i = 0; i<n - 1; i++) {
		if (v[i] < pivot) {
			mySwap(v[i], v[store++]);
		}
	}
    // 정렬된 값 바로 뒤 값과 pivot값 바꾸기.
	mySwap(v[store], v[n - 1]);

    // pivot index 값이 찾으려는 index 값과 같다면
	if (store == k) {
		return pivot;
	}
    // pivot index 값이 찾으려는 index 값 보다 크다면
    // 처음부터 pivot-1 사이의 배열에서 탐색시작 
	else if (store > k) {
		return MedianOfMedians(v, store, k);
	}
    // pivot index 값이 찾으려는 index 값보다 작다면
    // pivot+1 부터 끝까지의 배열에서 탐색시작
	else {
		return MedianOfMedians(v + store + 1, n - store - 1, k - store - 1);
	}
}
{% endhighlight %}