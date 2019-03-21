---
layout: post
title: Remote Client 영상 표출 과정
date: 2019-03-21 09:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## Live 555 Pipeline
 - source -> filter -> filter -> sink
 - source -> filter -> sink
 - source -> sink

## MediaRTSPSession 에서 RTSPClient 를 만든다.

## RTSPClient 는 다음의 Request 를 순서대로 카메라에 보낸다.
 - Describe >> Setup >> Play
 - Announce >> Setup >> Record

## 각 Request 후 다음의 작업을 수행한다.
 - AfterDescribe : subsession 만들기, subsession iterate 진행, Request Setup
 - AfterSetup : 코덱에 맞는 sink 만들기, RTSPSession 을 decoder Litener로 등록, Request Play
 - AfterPlay : ??

## OpenURL 이후 >> doEventLoop
 - doEventLoop >> SingleStep >> Decode
 - OnDecode 에 Frame 데이터 넘겨줌 >> RTSPSession Queue 에 Frame 쌓음.
 - IvsStream 에서 Session 의 frame 을 받아서 Queue 에 쌓음.
 - MultiScreen 에서 IvsStream 프레임을 받아서 Renderer::SetBitmap 호출.
 - Renderer 에서 image_bitmap_list 에 받는다.

## MultiScreen 이 IvsStream, D2DRenderer 를 갖는다.
 - stream 은 stream 마다 만들어진다.
 - renderer 는 MultiScreen 당 1개 만들어진다.