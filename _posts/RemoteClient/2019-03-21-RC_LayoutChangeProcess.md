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
 
## Renderer::SetLayout 하면 일어나는 일은?
 - row_count 와 column_count 값 바꿈.