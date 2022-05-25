---
layout: post
title: Keep Aspect Ratio 계산하는 과정 
date: 2019-03-25 16:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## 현재 영상 프레임의 ratio 를 구한다.
 - frame_ratio = frame.width() / frame.height()

## ~~뷰어 프레임 ratio 로 나눈다.~~
 - ~~screen_frame_ratio = frame_ratio / ratio(= 셀 가로 길이 / 셀 세로 길이)~~

## frame_ratio 와 ratio 같으면
 - image_rect 그대로 그린다.

## frame_ratio 가 더 크면( 옆으로 더 길쭉하면 )
 - UI 너비를 모두 이용한다.
 - 새로운 높이를 구한다.
    - ~~가로 길이가 줄어든 % 만큼 세로 길이를 줄인다.~~
    - ~~이 수치가 1 / screen_frame_ratio~~
    - ~~따라서, new_height = height / screen_frame_ratio~~
    - frame_ratio 만큼 높이를 변화시킨다.
    - 따라서, new_height = ui_height / frame_ratio
 - 그릴 image_rect 을 계산한다.

## frame_ratio 가 더 작으면( 위아래로 더 길쭉하면 )
 - UI 높이를 모두 이용한다.
 - 새로운 너비를 구한다.
    - ~~세로길이가 줄어든 % 만큼 가로 길이를 줄인다.~~
    - ~~이 수치가 screen_frame_ratio~~
    - ~~따라서, new_width = width * screen_frame_ratio~~
    - frame_ratio 만큼 너비를 변화시킨다.
    - 따라서, new_width = ui_width * frame_ratio
 -  그릴 image_rect 을 계산한다.
