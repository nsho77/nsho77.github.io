---
layout: post
title: MapView_JSON 데이터 Load, Save 과정 정리
date: 2019-04-16 10:15:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [Remote Client]
---
# Load
## 처음 프로그램 시작시
 - MapViewControl::OnCreateControl 호출.
 - LoadMaps() 호출.
    - map.json 파일 위치 찾기.
    - json 데이터 읽어서 List 형태로 만들기
    - MakeMapNode 호출

## MakeMapNode 하는 일은
 - RadTreeNode 생성

## Node 추가
 - MapTreeManager 에 root 있으면 root Node Collection에 추가.
 - root 없으면 treeView 에 추가.

# Save
## EditMode Switch 되면
 - MapViewControl::OnEditModeSwitchValueChanged() 호출.
 - SaveCurrentMap() 호출.

## SaveCurrentMap 에서 하는 일은
 - map.json 파일 위치 설정.
 - SaveMap 호출
    - name 매개변수로 MapJsonData 생성.
    - 파일이 이미 있으면, 읽어서 Map 에 저장. MapJsonData로 수정. 위치에 파일 저장.
    - 파일이 없으면, Map 에 추가. 위치에 파일 저장.