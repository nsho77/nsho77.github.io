---
layout: post
title: mfc-imageProcessing 화면에 흑백이미지를 함께 출력해보자
date: 2018-05-25 09:27:00 +0900
description: mfc imageprocessing 화면에 흑백이미지를 함께 출력해보자 # Add post description (optional)
img:  # Add image post (optional)
---

기존 ImageProcessing [작업](../mfcImageProcessing) 에서는 화면에 그림 하나를 띄웠다.
이번에는 화면에 원본그림 하나와 흑백이미지 그림 하나를 함께 띄우는 방법을 알아보자.

순서는 이렇다. 먼저 이미지를 읽는다. -> 이미지를 흑백으로 만든다. -> 원본과 흑백을 함께 띄운다.

현재 화면에 그림을 그리는 방법을 다르게 해야한다. 원본 이미지의 2배 만큼 길이를 잡고 왼쪽에는 원본을 오른쪽에는 흑백 그림을 띄운다.
이를 위해 SetDraw 함수를 변경해보자.