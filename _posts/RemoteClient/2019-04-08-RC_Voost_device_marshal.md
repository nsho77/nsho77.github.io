---
layout: post
title: VOOSTDEVICE 마샬링
date: 2019-04-08 09:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## VOOSTDEVICE 사용하기 위한 UI 환경
 - ICamera, IUnit 구현하는 VoostCamera Model. (Camera 와 같음)
 - VoostCamera 에 데이터를 채우고, Tree 와 정보교환 할 수 있는 VoostDeviceManager.
 - Voost.dll 함수 가져와 사용할 수 있게 만드는 VoostSDKProxy.
 
## Native 환경
 - Voost.dll 감싸는 VoostSDK.dll 생성.
 - VoostSDK.dll 에서 함수 표출.

## VOOSTDEVICE 가져오는 방법
 - Voost.dll VoostGetDevices 호출 할 수 있는 함수 생성. 표출.
 - 함수 마샬링 및 바인드.

## 마샬링 방법.
 - out IntPtr 로 Device 배열 가져오기, out UInt32 로 배열 사이즈 가져오기.
 - 위 매개변수 받고 Int32 return 하는 delegate 선언, 생성, Native 함수와 bind.
 - 함수 호출 후 가져온 디바이스 배열을 마샬링.

## VOOSTDEVICE 구조체 마샬링
 - CharSet 은 Unicode, Int 는 Int32, unsigend int 는 UInt32, wchar_t* 는 String으로 마샬링.

## VOOSTDEVICE 배열 마샬링
 - PtrToStructure<>() 이용.
 - 제네릭 사이즈 만큼의 포인터를 구조체로 변환.
 - 포인터에 VOOSTDEVICE 사이즈 만큼 더 하는 방식으로 배열 Iterate.
    
