---
layout: post
title: PThread vs Thread
date: 2019-03-19 09:05:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++, Programming Principles and Practice Using C++]
categories : [C++]
---
# std::thread vs posix thread
 - 어떤 플랫폼에서도 작동하길 원한다면 Posix Threads 를 사용해라.
 - Linux/gcc 만 사용하면 std::thread 좋다. 사용하기 좋은 인터페이스를 두루 갖추고 있다.
 - 현재는 std::thread 도 다양한 플랫폼을 지원한다.

# boost::thread
 - std::thread 랑 매우 비슷하다.

 참조 : https://stackoverflow.com/questions/13134186/c11-stdthreads-vs-posix-threads