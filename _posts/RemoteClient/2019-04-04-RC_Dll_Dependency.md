---
layout: post
title: Dll 의존성 해결하기
date: 2019-04-04 09:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## 문제가 발생한 상황
 - Voost.dll 을 이용하는 VoostSDK.dll 프로젝트 생성.
 - VoostSDK.dll 을 LoadLibrary 함.
 - 모듈을 찾을 수 없다는 메시지.

## 발생 이유
 - VoostSDK.dll 은 Voost.dll 에 의존적.
 - Voost.dll 은 seLive.dll, seDevInterface.dll 에 의존적.

## dll 의존성 해결방법
 - 의존성 있는 dll을 모두 .exe 와 같은 경로에 복사.

## C#, C++ Platform 판단 차이점
 - 32bit dll 이 64bit dll을 호출하면 안됨. 반대도 마찬가지.
 - C#은 컴퓨터의 환경으로 Platform 확인하고
 - C++ 은 빌드환경으로 Platform 정함.
 - 따라서 32bit 환경으로 빌드, 컴퓨터는 64bit, C# 이 빌드환경에
 - 따라 dll 을 가져오게 하면 빌드환경에 상관없이 64bit 만 가져옴.

## Platform 문제 해결방법
 - C++ 은 32bit 64bit 모두 미리 빌드.
 - C# 은 컴퓨터 환경에 따라 가져가게 만들기.