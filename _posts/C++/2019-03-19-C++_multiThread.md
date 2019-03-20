---
layout: post
title: 멀티스레드
date: 2019-03-19 09:00:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++, Programming Principles and Practice Using C++]
categories : [C++]
---
# 프로세스와 스레드의 차이점
 - 프로세스는 운영체제로부터 프로세서, 필요한 주소공간, 메모리 등 자원을 할당받는다.
 - 스레드는 프로세스가 할당받은 자원을 이용해 공유하면서 실행한다.

# 멀티스레드 프로그래밍 주의할 점
 - 스레드 안정성(Thread-safe) 중요
 - 프로그램 종료 전에 스레드를 먼저 종료하고 프로그램을 종료해야 한다.

 ![Multithread](https://wayhome25.github.io/assets/post-img/cs/thread.png)