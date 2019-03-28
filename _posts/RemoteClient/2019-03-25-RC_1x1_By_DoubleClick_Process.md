---
layout: post
title: Double Click 으로 화면 확대하는 과정 
date: 2019-03-25 10:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## MultiScreen::WndProc 가 MultiScreen::NotifyMouseDoubleClick 호출
 - Renderer::DoubleClick 호출.
 
## 아직 확대되지 않은 상황이라면??
 - 스트림 벡터를 돌면서, 선택한 타일과
 - 스트림이 가지고있는 타일이 같으면 스트림이 가진 타일을 prev_tile_index에 저장하고
 - max_channel_index 에 현재 스트림 인덱스를 저장한다.
 - 타일이 0인 스트림을 만나면 스트림 인덱스를 prev_first_channel_index 에 따로 저장한다.
 - 이후,
 - max_channel_index 의 tile_index 를 0으로,
 - prev_first_channel_index의 tile_index 에 prev_tile_index 를 저장한다.
 - SetLayout 을 호출한다.

## 이미 확대된 상황이라면??
 - 확대되기 전, 첫번째 채널이 연결되어있었으면,
 - 그 채널의 tile_index 를 0으로 만든다.
 - 현재 확대되어 있는 스트림의 tile_index 를 prev_tile_index 로 바꾼다.
 - 이전 행렬 갯수, lay_out_type 을 매개변수로 SetLayout 을 호출한다.
 - prev_tile_index, prev_first_channel_index 값을 초기화한다.