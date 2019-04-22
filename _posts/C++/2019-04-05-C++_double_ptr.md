---
layout: post
title: 더블포인터에 대해서
date: 2019-04-05 09:30:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++]
categories : [C++]
---

## 더블포인터란
 - 싱글 포인터를 가르키는 포인터

## 배열을 가리키는 포인터?
 - 더블 포인터는 배열을 가르키는 포인터가 될 수 있음.

## 이차원 배열?
 - 더블 포인터는 이차원 배열을 가르킬 수 없음.

## 더블 포인터의 두 가지 경우
 - A type 의 배열을 감싸는 배열
 - A type pointer 배열

## 포인터의 주소를 바꾸고 싶을 때
##### 싱글 pointer 를 사용한 경우
{% highlight cpp %}
void ChangePtrAddr(int* ptr)
{
    ptr = new int[3];
}

int* mainPtr = new int[5];
ChangePtrAddr(mainPtr);

{% endhighlight%}

 - ptr 이 Stack 생성 되고 mainPtr 값 복사됨.
 - ptr 에 새로운 주소 할당.
 - 그러나, 함수 빠져나오면서 ptr 소멸.
 - mainPtr 값은 바뀌지 않음.


##### 더블 pointer 를 사용한 경우
{% highlight cpp %}
void ChangePtrAddr(int** dbptr)
{
    *dbptr = new int[3];
}

int* mainPtr = new int[5];
ChangePtrAddr(&mainPtr);

{% endhighlight %}
 - dbptr 이 Stack 생성, mainPtr 주소가 복사됨.
 - dbptr 역참조 mainPtr 의 주소를 바꿈.
 - 함수 빠져나오면서 dbptr 소멸
 - mainPtr 값 바뀜.