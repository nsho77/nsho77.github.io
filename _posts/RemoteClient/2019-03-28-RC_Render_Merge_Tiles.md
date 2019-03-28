---
layout: post
title: Merge Tile 그리는 방법
date: 2019-03-28 15:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

merge tiles 값 셋팅 참조 [Tile Merge 하는 과정](/RC_Merge_Layout)

## span 값 셋팅
 - 라이브 스트림의 tile 이 inner_layout으로 등록되어있으면,
 - row_span 과 column_span 을 inner_layout 의 값으로 설정.

## allocated_tiles 설정
 - merged 된 타일 모두 돌면서 allocated_tiles index true 로 설정.

## 다른 스트림과 동일하게 그림
 - 화면, ratio, flip, name 등등 
 - span 값만 다름.

## merging 중인 상태 그리기
 - 현재 마우스 위치 이용,
 - start tile, end tile 구하고 --> row_span, column_span 구함.
 - merge 중인 모든 타일을 now_merging_tiles 에 저장.
 - 이후에 row, column 돌면서 now_merging_tiles 에 있는 타일에 색을 입힘.

## Border 그리기
 - tile 순회하면서,
 - inner_layout 에 순회 tile 있는지 찾음.
 - span 셋팅.
 - border 그리기.
 - inner_layout 이 가지고 있는 merged tile에는 border를 그리지 않는다.


## Default Logo 그리기
 - 현재 재생중이고
 - inner_layout 이 가지고 있는 merged tiles 에는 logo를 그리지 않는다.
 - border 와 같은 방법으로 span 값을 이용하여
 - Logo를 그린다.
