---
layout: post
title: "[번역글] AJAX Basics Explained By Working At A Fast Food Restaurant"
date: 2018-05-31 08:40:00 +0900
description: medium 에서 올라온 글을 번역했다. AJAX에 관한 내용 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, translate]
categories : [Algorithm & Data Structure]
---

> medium 에 올라온 [글](https://medium.freecodecamp.org/ajax-basics-explained-by-working-at-a-fast-food-restaurant-88d95f5fcb7a)을 번역했다.


## 패스트 푸드 음식점으로 AJAX 기본 설명하기

#### AJAX(Asynchronous JavaScript and XML) 은 서버 사이드에 대한 이해가 없다면 이해하기 어렵다.

웹 개발을 시작할 때 먼저 HTML, CSS, JavaScript, JQuery 를 배웠다. 이후에 Node.js 와 Ruby on Rails를 접했다.

다이나믹 웹 어플리케이션을 어떻게 만드는지 알고 싶어서 서버와 커뮤니케이트 하기 위해 AJAX 쓰는 방법을 알아야 했다. 2005 년에 만들어진 것 같은 정적인 페이지를 만들기 싫었다.

프론트엔드는 백엔드와 많이 달랐다. 난 GET POST requet를 이해하기 위해 노력했다.

그리고 마침내 패스트 푸드 레스토랑 예를 생각했다. 맥도날드, 버거킹 을 가봤다면 당신은 당신만의 GET과 POST request를 작성할 수 있다.

이 글을 읽기 위해서 당신은 초급 수준의 jQuery 지식이 필요하다.

## AJAX는 어떻게 생긴놈일까?

페이스북 페이지에 전체 페이지 리로딩 없이 댓글을 달 수 있다는 것을 알고 있는가? 이게 AJAX가 하는 일이다. AJAX로 모든 페이지를 리로딩 하지 않고 유저가 웹 어플리케이션과 소통할 수 있다.

좋아요나 댓글을 달때마다 페이지가 리로딩 된다고 생각하면 끔찍하다. 대신에, 페이스북은 좋아요, 댓글을 빨리 반영하고 당신이 계속 글을 읽을 수 있도록 해준다. 페이스북은 당신의 행동을 데이터 베이스에 저장하지만 당신의 행동을 방해하지는 않는다.

## AJAX는 왜 필요할까?

여기 몇 가지 예를 소개할 텐데 먼저 전체 흐름을 봐보자.

당신의 전체 웹 어플리케이션을 패스트푸드 음식점이라고 생각하자. 당신은 캐셔이고 가장 앞줄에 서 있는 사람이며 고객에게 request를 받아 처리한다.

![whole system]({{"https://cdn-images-1.medium.com/max/800/0*PBD--x73I4_zG0Kh."}})

위 그림을 보면 3개의 작업영역이 보인다.

1. 캐셔는 유저 request를 빠르게 처리한다.
2. 버거를 그릴위에 올리고 다른 음식도 요리한다.
3. 음식을 준비하여 종이가방이나 쟁반에 내놓을 팀이 필요하다.

AJAX가 없으면 당신은 처음부터 끝까지 음식이 준비되기까지 주문 하나만을 받을 수 있다. 주문을 받고 계산을 하고.. 주방에서 요리할 때까지 아무것도 안하고 기다린 다음, 음식 포장팀이 준비를 마칠 때까지 계속 기다린다. 이 모든 과정이 끝나야 다음 주문을 받을 수 있다.

![system without ajax]({{"https://cdn-images-1.medium.com/max/800/0*89PNzvIka550TPv2."}})

이런 시스템은 나쁜 유저 경험을 제공한다. 당신은 이것을 두고 패스트 푸드 라고 부르지 않을 것이다.

AJAX를 이용하면 비동기 프로세스 모델을 사용할 수 있다. 데이터를 요청하고 데이터를 보낼 때 전체 페이지를 로드하지 않아도 된다는 의미다. 보통 패스트푸드 음식점이 일하는 방식과 같다. 캐셔인 당신은 손님의 주문을 받고, 주방 팀에게 보낸다. 그리고 다음 손님의 주문을 받는다.

손님은 계속 주문할 수 있고 당신은 주방팀이 일할 때 앉아서 기다릴 필요가 없고 다른 사람을 기다리게 만들지도 않는다.

이 방법은 약간 복잡하다. 당신은 이제 음식점에 여러 전문분야가 생겼다. 주문은 각각 다른 속도로 처리된다. 그러나 사용자에게 더 나은 경험을 제공한다.

![system with ajax]({{"https://cdn-images-1.medium.com/max/800/0*716D3LoopXh8ILWC."}})

음식점에서 이렇게 일하는 걸 봤을 것이다. 한 사람은 튀김 기계에서, 다른 사람은 그릴에서 일한다. 주문이 들어오면 캐셔는 즉각 반응하고 바로 다음 주문을 받는다.

## POST Request 를 만드는 방법

이 컨셉을 이용해보자. 캐셔인 당신은 손님의 주문을 받아 주방으로 넘기고 그러면 나머지 사람들이 음식을 준비한다. 당신은 이를 POST request 로 할 수 있다.

실제 코드에서 POST request는 서버로 데이터를 보낸다. 데이터를 백엔드로 넘긴다는 것을 의미한다.

3가지 주요 파트가 있다:

1. A URL: request가 따라야 할 route 이다. 이후에 더 자세히 설명하겠다.

2. Data : 서버에 보내고 싶은 어떠한 파라미터.

3. Callback : request를 보내고 일어나는 것.

사람들이 패스트푸드 음식점에서 자주 주문하는 건 무엇일까?
다음 2가지 예를 보자.

1. 감자튀김

2. 감자튀김, 음료, 버거가 함께 있는 세트

위 두개의 프로세스는 서로 다르다. 감자튀김은 한 명이 다 요리할 수 있겠지만 콤보의 경우 다수의 사람이 필요하다. 따라서 이 두 경우는 다른 URL이 필요하다.

{% highlight javascript %}
$.post('/comboMeal')

$.post('/fries')
{% endhighlight %}

URL 이 같다면 request에 같은 로직이 적용된다. 이 튜토리얼 기사의 범위에 벗어나므로 궁금하면 더 파보시라.

다음은 data 이다. data는 request에 대해 더 많은 정보가 있는 object이다. combo meal URL 의 경우 우리는 아마 다음과 같은 정보가 필요할 것이다.

1. 메인 버거의 종류

2. 음료 종류

3. 가격

4. 고객 요구사항.

감자튀김의 경우, 다음과 같은 정보가 필요할 것이다.

1. 감자튀김의 크기

2. 가격

![system with ajax]({{"https://cdn-images-1.medium.com/max/800/0*6W8k6X4azQU9Jb3b."}})

combo meal 의 예를 보자. 치즈버거와 펩시콜라로 구성된 6 달러 짜리 세트다. JavaScript 에서는 아래와 같이 표현할 수 있다.

{% highlight javascript %}
let order = {
    mainMeal : 'cheeseburger',
    drink : 'Pepsi',
    price : 6,
    exceptions : ''
}

$.post('/comboMeal',order);
{% endhighlight %}

order 변수가 주문정보를 가지고 있다. POST request에 담아 보내면 우리 주방 팀원들이 combo meal 을 어떻게 구성할 지 알 수 있다.

이 코드는 아무때나 실행되면 안된다. request를 날릴 트리거 이벤트가 필요한데, 음식점에서 주문하는 건 웹페이지에서 주문 버튼을 클릭하는 것과 같다는 컨셉으로 다음과 같이 jQuery 를 이용해 click 이벤트를 만든다. 사용자가 click 하면 POST request를 날린다.

{% highlight js %}
$('button').click(function(){
    let order = {
        mainMeal: 'cheeseburger',
        drink: 'Pepsi',
        price: 6,
        exception: ''
    };

    $.post('/comboMeal', order);
})
{% endhighlight %}

마지막 파트. 주문을 보내면 캐셔는 손님에게 "다음손님" 이라고 말해야 한다. 이와같이 주문이 접수 되었음을 알리기 위해 콜백을 사용한다.

{% highlight js %}
$('button').click(function(){
    let order = {
        mainMeal: 'cheeseburger',
        drink: 'Pepsi',
        price: 6,
        exception: ''
    };

    $.post('/comboMeal', order, function(){
        alert('Next customer please!');
    });
{% endhighlight %}

## GET Request 는 어떻게 만들까?

우리는 주문을 받을 수 있는 상태다. 이제 음식을 손님에게 가져다줄 차례이다.

GET request 가 그와 같은 일을 할 수 있다. GET request로 서버에 데이터를 request 할 수 있다.(이 예에서는 주방) 주목할 사항: 지금 우리의 데이터베이스는 주문으로 가득찼다 음식이 아니라. GET request가 데이터베이스를 변경할 수 없다는 것은 중요한 차이점이다. 프론트 엔드에 정보만을 전달할 수 있다. POST request는 데이터베이스의 정보를 변경시킬 수 있다.

아래는 음식을 받을 때 받을 수 있는 질문 몇 가지이다.

1. 여기서 먹어요, 포장해 가세요?

2. 케첩이나 머스타드 소스같은 거 필요하세요?

3. 영수증 번호 뭐에요?

가족 먹을 콤보 3개를 주문했다고 해보자. 음식점에서 먹을 것이고 케첩이 필요하다. 영수증 번호는 191이다.

'/comboMeal' URL로 GET request를 만들자. URL은 POST request 와 같다. 그러나 이번에는 다른 데이터가 필요하다. 완전히 다른 종류의 request 이다. 같은 URL로 코드를 잘 정리할 수 있다.

{% highlight js %}
let meal = {
    location: 'here',
    condiments: 'ketchup',
    receiptID: 191
};

$.get('/comboMeal', meal);
{% endhighlight %}

![get request]({{"https://cdn-images-1.medium.com/max/800/1*v3wuuGaPDFsYr-pgR7a6bg.png"}})

트리거가 마찬가지로 필요하다. 손님이 캐셔의 질문에 답하면 request 가 트리거 된다. JavaScript 에는 묻고 답하기를 표현할 편리한 방법이 없어서 'answer' 버튼에 클릭이벤트를 생성하는 것으로 대신하겠다.

{% highlight js %}
$('.answer').click(function(){
    let meal = {
        location: 'here',
        condiments: 'ketchup',
        idNumber: 191
    }

    $.get('/comboMeal', meal);
});
{% endhighlight %}

![get request]({{"https://cdn-images-1.medium.com/max/800/0*dbcZP0FyqCa19uYK."}})

우리는 191번 주문의 3개의 콤보 세트를 받을 거기 때문에 콜백 함수가 필요하다. 콜백함수에서 data 파라미터를 통해 해당 정보를 받을 수 있다.

이 함수는 백엔드에서 주문 번호 191을 처리한 결과를 리턴한다. eat 이라는 이름의 함수명을 쓸건데, 결국 음식을 먹을 것이기 때문이다. 그러나 JavaScript에는 eat 함수가 없다.

{% highlight js %}
$('.answer').click(function(){
    let meal = {
        location: 'here',
        condiments: 'ketchup',
        idNumber: 191
    };

    // data contains the data from the server
    $.get('/comboMeal', meal, function(data){
        eat(data);
    })
})
{% endhighlight %}

마지막 제품인 데이터는 이론적으로 3개의 콤보 밀일 것이다. 백엔드에서 처리하는 방식에 따라 달라진다.

![get callback]({{"https://cdn-images-1.medium.com/max/800/0*CmjCchSTgQN7L6Bg."}})


## 다른 예 찾아보기

다른 예들도 찾아보자. (끝)