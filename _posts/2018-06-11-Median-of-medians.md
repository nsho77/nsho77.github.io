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

1. M = select({x[i]}, n /10) 를 처음 호출할 때, {x[i]}는 다음의 숫자를 가지고 있을 것이다. 50 45