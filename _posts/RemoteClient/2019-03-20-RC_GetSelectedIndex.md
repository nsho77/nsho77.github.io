---
layout: post
title: GetSelectedIndex 얻는 방법
date: 2019-03-20 13:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---
- IvsScreenControl.SelectedIndex 는 Property 이다.
- 마샬링을 이용해 MultiScreen::GetSelectedIndex 호출한다.
- D2DRenderer::GetSelectedStreamIndex로 selected_index 가져온다.

## selected_index 값을 변경시키는 방법은?

