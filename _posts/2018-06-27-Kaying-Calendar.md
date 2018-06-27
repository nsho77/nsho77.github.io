---
layout: post
title: 카잉 달력 문제 풀기
date: 2018-06-27 09:30:00 +0900
description: 백준문제 풀기 # Add post description (optional)
img:  # Add image post (optional)
tags: [Algorithm, Code]
---

> 백준에 있는 카잉달력 문제를 풀어보았다. [문제링크](https://www.acmicpc.net/problem/6064)

문제의 조건은 이렇다.
M 과 N 보다 작거나 같은 두 개의 자연수 x, y를 가지고 각 년도를 <x:y> 로 표현한다.
<x:y> 의 다음 해를 <x`:y`> 라고 하자. 만일 x<M 이면 x` = x+1 이고 y <N 이면 y`=y+1 이다.
그렇지 않으면 x` = 1, y`=1 이다. <M : N> 은 마지막 년도이다.
예를 들어 M = 3 , N = 2 이면

1 번째 년도 : 1:1
2 번째 년도 : 2:2
3 번째 년도 : 3:1
4 번째 년도 : 1:2
5 번째 년도 : 2:1
마지막 년도 : 3:2

M과 N의 최소공배수가 마지막 해가 된다.

내가 푼 방식은 이렇다.
x에 맞는 해를 찾는다 그때의 y를 _y 라고 하자. 그리고 x 만큼 더하면서 y == _y 이면
그때의 해를 반환하고 M 과 N의 최소공배수 까지 못찾으면 -1 을 반환한다.

두 수 a, b의 최소공배수는 (a*b)/최대공약수(a,b) 임을 이용하고
최대공약수(a,b) = 최대공약수(b,a mod b), if a mod b == 0 이면 최대공약수(a, b) = b인
성질을 이용한 유클리드 호제법을 이용해 최대공약수를 구한다.

{% highlight cpp %}
#include <stdio.h>
void kaying(int M, int N, int x, int y)
{
    // 유클리드 호제법을 이용, 최대공약수를 구한다.
    int a = M, b=N;
    int modValue = a % b;

    while(1)
    {
        if(!modValue)
            break;
        a = b;
        b = modValue;
        modValue = a%b;
    }

    int gcd = b;
    // 최소공배수를 구한다.
    int lcm = (M*N) / gcd;

    // x 값에 해당하는 _y 값을 찾는다.
    int _y = (x%N)==0 ? N : x%N;

    // 몇 번째 해인지 저장한다.
    int year = x;

    // 최소공배수까지 _y == y 인 값을 찾는다.
    while(1)
    {
        if(year > lcm)
        {
            year = -1;
            break;
        }
        if(_y == y)
            break;

        _y = _y +M;
        _y = (_y%N==0) ? N : _y%N;
        year = year + M;
    }

    printf("%d\n", year);

}
{% endhighlight %}

아래는 다른 분이 푼 소스이다. 보면서 나랑 무엇이 다른지 비교해보자.

{% highlight cpp %}
#include <cstdio>

// 최소공배수를 구한다.
int lcm(int m, int n)
{
    int z,a,b;
    a = m; b= n;
    while(1)
    {
        z = a%b;
        if(z==0) break;
        a = b; b = z;
    }
    return (m*n)/b;
}

int main()
{
    // t번 문제를 푼다. (문제의 조건임. 위에서는 구현안함.)   
    int t; scanf("%d", &t);
    while(t--)
    {
        int M,N,x,y;
        scanf("%d%d%d%d",&M,&N,&x,&y);
        // M, N의 최소공배수를 구한다.
        int mm = lcm(M,N);
        // x와 y값을 비교해서 작은 값에 m, n을 더하고
        // x== y가 될 때의 x를 반환하면 된다.
        // x<= 최소공배수(M,N)까지 반복한다.
        while(x!==y && x<=mm)
        {
            if(x<y) x+=m;
            else y+= n;
        }
        if(x!=y) printf("-1\n");
        else printf("%d\n",x);
    }

    return 0;
}
{% endhighlight %}