---
layout: post
title: 정적라이브러리, 동적라이브러리 비교
date: 2019-03-27 11:30:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++]
categories : [C++]
---
## 정적라이브러리 (lib)
 - 정적라이브러리를 프로젝트에 포함시키면
 - 링크 과정에서 프로젝트에 포함되어 실행파일이 만들어진다.

## 동적라이브러리 (dll)
 - 동적라이브러리를 프로젝트에 포함시키면
 - 링크 과정에서 포함되지 않는다.

## dll 사용 장점
 - 메모리 절약
 - 실행 이미지크기 작아짐
 - 교체 및 디버깅 용이
 - 컴파일 시간 줄어듦.

 ## 외부 라이브러리 포함 방법
  - lib 인 경우, 링크 디렉터리, 추가 종속성, 헤더파일 위치 설정
  - dll 인 경우, 링크 디렉터리, 추가 종속성, 헤더파일 위치 설정 후 실행파일과 같은 디렉토리에 dll 위치시킴.

  