---
layout: post
title: MapView_Live 영상 나오는 과정 정리
date: 2019-04-15 09:40:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## 영상 Play 과정
 - MapItemCollection에서 이벤트 달아줌.
 - GraphicMapContorl 핸들러 호출.
 - MapViewControl 핸들러 호출.

## MavViewControl 핸들러에서 하는 일?
 - 핸들러 이름 : OnMapControlItemDoubleCliked(object sender, ItemEventArgs e)
 - e.Item.Key 와 deviceManager Camera list Cam ID 와 비교해서 카메라 Get.
 - e.Item 이용해서 IvsScreenControl 이 표시될 좌표 Set.
 - PlayCamera 호출.

## PlayCamera 하는 일?
 - 다른 카메라 스트림 모두 중지.
 - IvsScreenProxy 이용 Native 함수 StartPlay 호출.

## StartPlay 했을 때 Stream Set되는 방법?
 - <Uid, MultiScreen> map 에 IvsScreenControl 이 저장.
 - MultiScreen 에서 Stream 관리.

## Stop 했을 때 Live, Map 스트림 함께 Stop 되는지?
 - 아님.
 - Screen 을 저장하는 Uid 가 다름.