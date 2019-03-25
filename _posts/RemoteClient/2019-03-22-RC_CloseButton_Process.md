---
layout: post
title: Mouse Event 가 Close Button 에 영향을 주는 과정 
date: 2019-03-22 14:50:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## MultiScreen::WndProc 가 MultiScreen::NotifyMouseMove 호출
 - SetMousePoint -> GetTileIndexByLocation 호출.
 - over_tile_index 셋팅.

## renderer 가 해당 tile_index 에 닫기 이미지를 Draw.
 - GetButtonRect() 가 image_rect, button_rect 셋팅.
 - DrawImageButton() 이 button_rect 크기의 타원을 그리고
 - mouse_location 판단하여 타원에 색을 칠하거나 칠하지 않는다.
 - X 이미지를 그린다.
