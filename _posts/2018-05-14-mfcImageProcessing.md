---
layout: post
title: mfc-imageProcessing 을 훑어보자~!
date: 2018-05-14 19:20:00 +0900
description: mfc imageprocessing 하나하나 풀어보자 # Add post description (optional)
img:  # Add image post (optional)
---

mfc 로 프로젝트를 생성하면 다음과 같은 소스파일이 생성된다.
* ImageProcessing.cpp
* ImageProcessingDoc.cpp
* ImageProcessingView.cpp
* FileView.cpp
* MainFrm.cpp
...

이중에서 가장 중요한 소스는 위부터 3개다. 위 3개에 나머지 소스가 종속되어있다고 보면 된다.

ImagaProcessing.cpp 은 전체적인 흐름을 담당하고 ImageProcessingDoc.cpp 은 메뉴, 문서 등을 담당하며 ImageProcessingView.cpp 은 이미지가 보이는 화면을 담당한다.


먼저 ImageProcessingView.h를 보자 ImageProcessingView.cpp 에서 정의할 클래스의 변수, 메서드를 볼 수 있다.

{% highlight cpp %}
class CImageProcessingView : public CView
{
    ...
}
{% endhighlight %}

class 선언으로 class를 만들 수 있다. ':' 는 CView를 상속받는다는 의미다.

{% highlight cpp %}
class CImageProcessingView : public CView
{
protected:
    CImageProcessingView();
    DECLARE_DYNCREATE(CImageProcessingView);

private:
    ...

public:
    ...
}
{% endhighlight %}

protected, private, public 등의 접근 제한자로 상황에 알맞은 변수, 함수를 선언할 수 있다.

> 질문 : protected 안에 CImageProcessingView()는 무엇일까??