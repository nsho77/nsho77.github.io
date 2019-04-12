---
layout: post
title: MapView_ZoomIn_Out 따라가보자
date: 2019-04-12 09:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## ZoomIn
 - 마우스 휠 이벤트 -> drawingObject.Scroll()호출
 - 이벤트 Delta 값 확인하고 ZoomIn 호출
 - ZoomIn 단계는 0 에서 0.5 사이가 되도록 조절
   - Zoom Set 하면 SetBoundingRect 호출 -> 다시 그림.
 - PointToOrigin 호출

## PointToOrigin 하는 일은?
 - ZoomIn 했을 때 보여줄 왼쪽 위 모서리 좌표 계산
 - 현재 모서리 좌표와 늘어난 지도 사이즈 매개변수
 - 현재 모서리 좌표가 새로운 모서리 좌표의 가운데에 오게끔 계산
   - 컨테이너 상단 가운데 좌표얻기.
   - 늘어난 지도에서 좌표 구하기.
   - 좌표가 컨테이너 상단 가운데에 오게 계산.

## SetBoundingRect 호출
 - 모서리 좌상단 좌표 저장.
 - 현재 지도 이미지 사이즈 저장.

## AvoidOutOfScreen 호출
 - 모서리 좌상단 좌표가 좌표범위를 벗어났는지 판단.
   - 넘어가면 0으로 Set.
 - 모서리 좌성단 좌표 + 컨테이너 넓이가 지도 좌표를 넘어가는지 판단.
   - 넘어가면 지도 사이즈 - 컨테이너 사이즈 좌표로 Set.

## ZoomOut
 - Zoom에서 step보다 작거나 같은 수 빼서 Zoom Set.