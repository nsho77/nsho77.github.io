---
layout: post
title: lvalue, rvalue 알아보자
date: 2019-04-29 09:00:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++]
categories : [C++]
---

## Lvalue, Rvalue 는 무엇?
 - Lvalue, Rvalue 는 expression 에서 정의된다. object 와 관계 없다.
    - double d; // d 는 double object이고 l,r value 의미없다.
    - d = 3.1415*2; // expression 에서 lvalue, rvalue 가 정의된다.
 - Lvalue 는 표현식에서 object(메모리를 가지고 있음)임.
 - Rvalue 는 표현식에서 Lvalue 를 제외한 표현식임.

## Lvalue, Rvalue 예들...
 - Numeric, character literal 은 Rvalue.
 - enum Value 들 Rvalue.
    - enum Color {red, green, blue} // red, green, blue 는 Rvalue
 - & 참조연산은 Lvalue 만 사용가능, 연산결과는 Rvalue
    - int n, *p; p = &n;
 - 포인트* 연산은 결과로 Lvalue 반환.
 - pre-increment (++nCount) 결과로 Lvalue 반환.
 - 리턴 type 이 reference 인 경우 함수 콜은 Lvalue 이다.
 - 참조는 그냥 이름. 따라서 Rvalue 에 묶인 참조 그 자체는 Lvalue 이다.
 - Rvalue 는 임시적이고, 해당 메모리에 접근하는 것이 권장되지는 않지만 접근은 가능하다.
 - post-increment (nCount++) 결과로 Rvalue 반환.

 ## 정리하자면..
  - 표현식이 끝나도 존재하는 것은 Lvalue.
  - 표현식이 끝나면 사라지는 임식적인 값은 Rvalue.


참조 [https://www.codeproject.com/Articles/313469/The-Notion-of-Lvalues-and-Rvalues](https://www.codeproject.com/Articles/313469/The-Notion-of-Lvalues-and-Rvalues)