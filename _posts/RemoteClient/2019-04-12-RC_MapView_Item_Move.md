---
layout: post
title: MapView_Item_Move 하는 법
date: 2019-04-12 10:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---
1. 
## MapItemCollection 에서 Event Attach
 - GraphicMapControl에서 Add 등 할때 호출
 - MouseMove Event -> OnItemMoved 호출 -> SetAbsoluteLocation 호출

## SetAbsoluteLoaction 하는 일
 - mapItem 중심 좌표를 현재 지도 사이즈 대비 비율로 변환
 - item.AbsoluteLocation 에 저장.

2. 
## ControlMoverOrResizer 에서 Event Attach
 - Initialize 로 등록
 - MoveControl 호출

## MoveControl
 - 컨트롤이 이동해야 할 좌표구하기.
 - 좌표가 화면을 벗어나는지 체크.
 - control 새로운 Location Set.