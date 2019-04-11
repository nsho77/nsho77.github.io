---
layout: post
title: MapView_UpdatePanels 따라가보자
date: 2019-04-10 09:01:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## UpdatePanles 위치는?
 - GraphicMapControl >> Map, PictureBox, Panel 그리는 곳.

## UpdatePanels 하는 일은?
 - panel 크기에 맞게 축소된 지도 bitmap 그리기.
 - 축소된 지도 bitmap 약간 투명하게 만들기.
 - 보고 있는 지도 위치에 맞게 panel 에 박스 그리기.

## 축소된 지도 투명하게 하여 그리는 방법은?
 - Bitmap 변수 만들고 이 변수에서 Graphic 얻음
 - Graphic::DrawImage 이용, 지도 그림.
 - GetTranslucentBitmap 호출하여 지도 투명하게 만듦.
    - [이미지 투명하게 만드는 법](/)

## 박스 그리는 방법?
 - width, height ratio 구함.
    - panel width / CurrentSize width
    - CurrentSize 는 원래 지도 사이즈에 * zoom 한 것.
 - ratio 이용하여 boxWidth, height, positionX, Y 구함.
    - [Bouding Rect Position 잡는 방법](/RC_MapView_Bounding_Rect_Position)
 - zoom 되어 있으면 박스 그림.
