---
layout: post
title: Channel Pointer 얻는 방법
date: 2019-03-20 09:15:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---
1. channel_infos vector 에서 index 번째 pointer 를 얻어온다.

## 어떻게 channel_infos vector 를 Setting 하는 거지?

- Renderer 만들어 질때 max_channel 만큼 공간을 만들고,
- MultiScreen::StartPlay 에서 Set 한다.
- channel_infos 의 순서와 tile index 는 상관없다.

## 각 channel_info 에 tile_index 를 setting 하는 방법은?

 - D2DRenderer::AddChannel
 - representative_tile_index 가 있으면 tile index = representative_tile_index
 - 없으면, tile_checker 에서 비어있는 index

 ## 2x2 화면에서 5번째 camera 연결하면 재생 되는 이유?

