---
layout: post
title: 함수명 뒤에 오는 const
date: 2019-03-20 09:05:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++, Programming Principles and Practice Using C++]
categories : [C++]
---
# ex) ChannelInfo* D2dRenderer::GetChannelPtr(const int index) const {}
 - 멤버함수에서 사용.
 - 모든 멤버 변수를 const 처럼 사용.
 - 이 함수에서는 멤버변수를 수정하지 않겠다는 표시.

 참조 : https://kldp.org/node/42466