---
layout: post
title: Tile Merge 하는 과정 
date: 2019-03-28 10:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## Renderer::merge_start_tile_index 설정


## 1.new index 설정
 - start 와 end tile 을 구분.
 - 이 둘중 더 왼쪽, 더 위쪽에 있는 타일을 찾음.
 - 이 타일 index 를 new index 에 저장.
 - new index, row_span, column_span 을 매개변수로 MergeLayout 호출.

## 2.Renderer::MergeLayout() 동작
 - inner_layouts
   - merge 가 시작되는 tile index 를 key, Layout 을 value로 가지고 있는 map.
 - merged_tiles
   - tile_index 를 index로 갖고, 해당 타일이 merge tile 의 일부인 경우,
   - merge_start_tile_index 를 값으로 갖는 vector.
 - new index 부터 row_span, column_span 만큼 타일을 돌면서
 - 해당 타일이 inner_layout 에 있는지 확인.

## 3.inner_layout에서 못 찾으면?
 - merged_tiles 에서 merge_start_tile_index 를 찾고,
 - 이를 key로 하여 inner_layout 에서 찾음.
 - 찾으면 4번 진행

## 4.inner_layout에서 찾으면?
 - key(찾은타일)의 레이아웃 정보를 이용.
 - merged tile 들 unmerge.
 - key를 inner_layout 에서 제거.

## 5.merged_tiles 에 start tile 정보 세팅

## 6.inner_layout 에 ChannelLayout 정보 세팅

## 7.merge area 에서 다른 스트림이 live 중이라면
 - 해당 스트림의 tile_index = -1.

## 8.Renderer::ArrangeTiles() 호출
 -  [ArrangeTiles() 가 하는 일](/RC_ArrangeTiles_Process)


## 9.FireLayoutChangeEvent(row_count, column_count, layout_type) 호출
 - (참고 -- [Layout Change 되는 과정](/RC_LayoutChangeProcess))
 - UpdateAllDecodersFrameSize() 호출
 - RaiseLayoutChanedEvent() 호출

## 10.Renderer 에서는 merge tile 을 어떻게 그릴까?
 - [Merge Tile 그리는 방법](/RC_Render_Merge_Tiles)