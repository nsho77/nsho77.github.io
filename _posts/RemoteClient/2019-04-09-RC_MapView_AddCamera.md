---
layout: post
title: [MapView]AddCamera 따라가보자
date: 2019-04-09 18:01:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## Map View 카메라 리스트를 더블 클릭하면
 - MapViewControl::AddCamera 호출
 - GraphicMapControl::SelectedItem Set
 - PictureBox::IsSelected Set
 - PictureBox::Validate 호출
 - PictureBox::OnPaintBackground()
    

## PictureBox::OnPaintBackground() 에서
 - GraphicMapControl::OnPaintBackground 호출

## GraphicMapControl::OnPaintBackground 하는 일
 - 지도 Bitmap 그리기

## PictureBox::OnPaintBackground, OnPaint 하는 일
 - PictureBox Image 그리기
 - Text 그리기
    