---
layout: post
title: 부녀회장 될 테야 문제 풀기
date: 2018-05-09 13:00:00 +0900
description: 백준문제 풀기 # Add post description (optional)
img:  # Add image post (optional)
tags: [Algorithm, Code]
categories : [Algorithm & Data Structure]
---

> 백준에 있는 부녀회장이 될테야 문제를 풀어보았다. [문제링크](https://www.acmicpc.net/problem/2775)

재귀적으로 먼저 풀어보고 이를 사용하지 않고 풀어보았다.

{% highlight cpp %}
// k 층 n 호에 몇 명이 살고 있는지 알아낸다.
int countPeople(int k, int n)
{
    int rCnt = 0;
    if(k == 0)
        return n;

    for(int i=1; i<=n; i++)
    {
        rCnt += countPeople(k-1,i);
    }

    return rCnt;
}
{% endhighlight %}

다음은 재귀 함수 사용하지 않은 풀이
{% highlight cpp %}
int countPeople(int k, int n)
{
    // 최대 15층 14호까지 들어올 걸 알고 있다.
    // 15*14 배열을 선언한다.
    // 2차원 배열로 해도 되지만 이를 1차원으로 바꿔서 사용해보자.
    int arr[15*14] = {0};

    // 0층 1호부터 인자의 층과 호까지 계산하여 채워넣는다.
    // 배열의 인덱스는 0부터 시작이지만 방은 1호가 처음임을 주의해야 한다.
    for(int i=0; i<15; i++)
    {
        for(int j=0; j<14; j++)
        {
            if(i==0)
                arr[14*i+ j] = j+1;
            else if(j==0)
                arr[14*i + j] = 1;
            else
                arr[14*i + j] = arr[14*i + (j-1)] + arr[14*(i-1)+j];

            // k 층 n 호에 저장된 값을 return 한다.
            if(i == k && j == n-1)
                return arr[14*i +j];
        }
    }
        

}
{% endhighlight %}