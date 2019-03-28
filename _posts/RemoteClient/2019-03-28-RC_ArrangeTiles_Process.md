---
layout: post
title: ArrangeTiles() 가 하는 일 
date: 2019-03-28 14:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

- 타일체커를 만들고
  - merged_tile 이 있다면, 타일체커에 체크.  

- 라이브 스트림 돌면서
  - tile_index 를 타일체커에 체크.  

- unallocated_channels 를 만들고
  - 라이브 스트림 돌면서
  - tile_index 가 -1 인 스트림을 unallocated_channels 에 저장.  

- unallocated_channel 스트림을 돌면서
  - 빈 tile_index 를 스트림 tile_index 에 할당.  
