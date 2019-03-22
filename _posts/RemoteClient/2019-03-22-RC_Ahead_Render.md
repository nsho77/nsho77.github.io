---
layout: post
title: Render 함수 콜 앞에서 Direct2D 가 하는 일
date: 2019-03-22 13:40:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## ID2D1Multithread::Enter()
 - D2D API critical section 진입.

## CreateDeviceResource()
 - __자주 쓸 GDI Resource 생성.__
   - render_target_handle 생성. (window에 그릴 target)
   - 자주 쓸 brush 생성.
   - 자주 쓸 Bitmap Image 생성.
