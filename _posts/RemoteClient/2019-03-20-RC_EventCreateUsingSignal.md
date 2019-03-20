---
layout: post
title: boost::signals2 로 이벤트 만들기
date: 2019-03-20 16:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---
- signal 생성 -> 함수 Connect -> call Signal 순서.

## 예를 들어 Selected Screen Index 가 변경되었을 때 발생하는 이벤트
- D2DRenderer 에서 signal 생성하고, 
- signal 과 함수를 Connect 하는 함수 정의.
- MultiScreen 에서 signal 과 subscribe 함수가 실제 연결.
- D2DRenderer::SetSelectedTileIndex 에서 Call Signal.
- MultiScreen::RaiseSelectedIndexChangedEvent 에서 넘겨받은 콜백을 호출.

