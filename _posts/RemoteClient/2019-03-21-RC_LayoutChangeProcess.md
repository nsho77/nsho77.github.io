---
layout: post
title: Layout Change 되는 과정
date: 2019-03-21 16:15:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## Layout 변화 시키는 버튼 ( 2x2 ...) 클릭하면
 - LiveViewContorls::OnLayoutToggleStateChanged 호출 -> SetScreenLayout 호출 ->
 MultiScreen::SetLayout 호출 -> Renderer::SetLayout 호출 -> LiveViewControls::OnScreenLayoutChanged 호출

## LiveViewControls::OnLayoutToggleStateChanged 호출하면 일어나는 일은?
 - 이벤트 발생지역이 어떤 버튼인지 확인하고
 - 버튼의 해당 레이아웃을 매개변수로 SetScreenLayout() 을 호출함.

## Renderer::SetLayout 하면 일어나는 일은?
 - row_count 와 column_count 값 바꿈.

## LiveViewControls::OnScreenLayoutChanged 호출하면 일어나는 일은?
 - LayoutChangedEventArgs 확인하고
 - 버튼을 값에 맞게 토글시킴.