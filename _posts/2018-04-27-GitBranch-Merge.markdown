---
layout: post
title: Git 브랜치와 Merge 기초
date: 2018-04-27 15:45:00 +0900
description: Git 의 공식 설명서를 읽고 정리해 보았다. # Add post description (optional)
img: workflow.jpg # Add image post (optional)
tags: [Programming, Git] # add tag
---
깃의 [다음 페이지](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) 에서 상세한 설명을 볼 수 있습니다.<br />
velopert 님의 튜토리얼 중 git checkout 하는 부분이 있어서 관련 내용을 찾아보던 중 정리한 내용이다.  

###  브랜치란?
head가 가리키는 작업영역이다. 기본적으로 master 브랜치에서 작업하고 이 branch 와 다른 작업을 하고 싶을 때 새로운 branch를 생성해 작업 할 수 있다.<br />
작업하는 branch를 바꾸면 head 가 해당 branch를 가리킨다.<br />
다음은 branch를 생성하는 명령어이다. branch 명은 iss53으로 하자
```bash
$ git branch iss53
```
다음은 iss53으로 head 가 가리키는 영역을 바꾸는 명령이다.
```bash
$ git checkout iss53
```
branch iss53에서 commit을 하면 다음 그림과 같은 상태가 된다.
<img src="https://git-scm.com/book/en/v2/images/basic-branching-3.png" alt="make branch">

그런데 master 브랜치에서 해결해야 할 이슈가 생겼다면 어떻게 해야 할까? 일단 master 브랜치로 checkout 해야 한다.
이슈를 해결하고 commit 하기 위해서 새로운 브랜치 hotfix를 만들고 commit 한다. 그러면 아래와 같은 상태가 된다. <br />
<img src="https://git-scm.com/book/en/v2/images/basic-branching-4.png" alt="Hotfix branch based on `master`.">

master 브랜치로 와서 hotfix 브랜치의 내용을 merge 하여 수정 사항을 반영한다. hotfix는 master 브랜치를 base로 하기 때문에
merge 를 할 경우 단순히 master 가 가리키는 커밋이 hotfix가 가리키는 커밋으로 바뀌게 된다. 이를 master 브랜치가 hotfix로 fastforward 되었다고 한다.<br />
<img src="https://git-scm.com/book/en/v2/images/basic-branching-5.png" alt="`master` is fast-forwarded to `hotfix`.">

branch 를 지우는 명령은 다음과 같다.
```bash
$ git branch -d  hotfix
```
hotfix branch와 master branch 가 같은 commit을 가리키고 있으므로 hotfix 브랜치를 지웠다.<br />
그리고 이전에 이슈를 해결하던 branch로 돌아와 문제를 해결한 뒤 commit을 하면 다음과 같이 된다.<br />

<img src="https://git-scm.com/book/en/v2/images/basic-branching-6.png" alt="Work continues on `iss53`.">

모든 이슈가 잘 해결 되었고 해결한 이슈를 master 브랜치로 merge 하고 싶다면 master 브랜치로 돌아와 iss53을 merge 하면 된다.
```bash
$ git checkout master
$ git merge iss53
```
merge in 객체가 merge into 의 자식이었다면 앞에서처럼 fastfoward가 진행되었겠지만 지금 상황은 다르다.<br />
이럴 경우 git 은 master 브랜치와 iss53 브랜치의 common ancestor를 찾아 이 둘 branch와 3-way-merge를 진행한다.<br />
fastfoward 와는 달리 git은 아래 그림과 같이 새로운 commit 객체를 생성한다.<br />

<img src="https://git-scm.com/book/en/v2/images/basic-merging-2.png" alt="a merge commit" >

만약 두 branch의 같은 파일이 다르게 변경 되었다면 한쪽으로 변경하든지, 사용자가 직접 수정해야 한다. 그리고 commit을 해야 merge 가 완료된다.
conflict file 은 git status 명령을 통해 확인할 수 있다. 입력하면 다음과 같은 message를 볼 수 있다.
```bash
$ git status
On branch master
You have unmerged paths.
  (fix conflicts and run "git commit")

Unmerged paths:
  (use "git add <file>..." to mark resolution)

    both modified:      index.html

no changes added to commit (use "git add" and/or "git commit -a")
```
index.html 이 충돌하므로 commit 할 내용을 정하고 commit 하면 끄읕~!!

git의 default merge tool 과 다른 툴을 쓰고 싶다면
```bash
$ git mergetool
```
을 입력하면 된다.