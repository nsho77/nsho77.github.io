---
layout: post
title: c# (WPF) Data Binding 에 대해서
date: 2019-06-05 09:00:00 +0900
description: Design Pattern # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Design Pattern]
---

## 데이터 바인딩 구조
 - 바인딩 대상 : UI 요소
 - 바인딩 소스 : 데이터
 - 바인딩 개체 : 대상과 소스를 연결

## 바인딩 개체가 제공하는 바인딩 방식
 - OneWay : 소스에서 대상으로만 데이터가 전달.
  ex) TextBox 의 Text 속성(바인딩 대상)에 바인딩 된 string 객체(바인딩 소스).
 - TwoWay : 소스, 대상 양방향 데이터 전달.
 - OneWayToSource : OneWay 반대 방향
 - OneTime : 한번만 소스가 대상을 초기화. 이후 변경 안됌.

## UpdateSourceTrigger
 - TwoWay 방식에서 대상에서 소스로 전달되는 과정
 - LostFocus : UI 요소가 포커스를 잃었을 때 바인딩 소스를 업데이트 한다.
 - PropertyChanged : UI 요소의 바인딩 된 속성 값이 변경될 때 소스를 업데이트 한다.
 - Explicit : 애플리케이션에서 명시적으로 UpdateSource 호출 할 때 업데이트 한다.


## INotifyPropertyChanged 인터페이스 구현
 - 소스의 변화를 대상에 전파하기 위해
 - PropertyChangedEventHandler 이벤트 객체를 통해 프로퍼티 값이 변경되었다는 것을
  UI 요소에 알림.