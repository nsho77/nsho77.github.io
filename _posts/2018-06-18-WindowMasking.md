---
layout: post
title: mfc-imageProcessing window masking을 해보자
date: 2018-06-18 08:30:00 +0900
description: mfc imageprocessing window masking 작업을 알아보자 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, imageProcessing]
---

이번에 할 것은 3x3, 5x5 사이즈의 윈도우 마스크를 만들어 이미지에 적용해보는 것이다.

먼저 마스크의 값과 이미지의 값을 곱한뒤 더하는, 즉 평균값을 넣는 마스크 작업을 해보자.

> ImageProc.h
{% highlight cpp %}

{% endhighlight %}