---
layout: post
title: Bouding Rect Position 잡는 방법
date: 2019-04-11 10:30:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## DrawingObject.boundingRect ??
 - 화면에서 보여주는 지도의 정보를 가지고 있음.
 - boundingRect.Width, Height 는 확대된 지도의 전체 크기정보.
 - boundingRect.X, Y 는 화면 왼쪽 위 모서리 좌표임.
   - 오른쪽 아래로 갈 수록 값이 작아짐.
   - 지도의 왼쪽 위 모서리 좌표는 (0,0).

## boundingRect Position 언제 Set?
 - Image Set 될때 (0,0) 으로 초기화.
 - Drag 할 때.

## Drag 할 때 어떻게 Set?
 - point : GraphicMapControl에서 마우스가 클릭 한 화면 좌표.
 - dragPoint : point - boundingRect position 한 좌표.
   - boundingRect position 기준인, 상대 좌표.
 - point - dragPoint : boundingRect position.
 - boundingRect x + ContainerWidth 가 BoundingRect.Width 보다 작으면 in panel.
   - boudingRect x 에 새로운 boundingRect x 저장
 - point.x - dragPoint.x 가 0 보다 크면 boundingRect x 0 으로 저장
 - boundingRect x + ContainerWidth 가 BoundingRect.Width 보다 크면
   - boundingRect x  = BoundingRect.width - ContainerWidth
 - height 도 마찬가지.
 - SetBoundingRect 호출해서 Set.