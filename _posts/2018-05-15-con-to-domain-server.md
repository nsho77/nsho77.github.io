---
layout: post
title: 도메인과 서버연결 과정을 살펴보자
date: 2018-05-15 15:30:00 +0900
description: 도메인과 서버연결 과정을 살펴보자 # Add post description (optional)
img:  # Add image post (optional)
---

도메인 구매 업체에서 도메인 www.GoSeongho.co.kr 을 구매하고 이미 다른 도메인이 연결된 나의 서버에 연결하려면 어떻게 해야할까?

1. 서버설정(아파치)를 수정한다.
2. 도메인 네임서버(DNS)가 내 서버를 가리키게 만든다.


## 아파치 설정을 변경
아파치 설정파일 httpd.conf에 다음과 같은 항목을 추가한다.

{% highlight apache %}
<VirtualHost 182.162.21.119:80>
        ServerAdmin 서버관리자정보
        ServerName www.GoSeongho.co.kr
        ServerAlias GoSeongho.co.kr
        DocumentRoot /www/seongho
</VirtualHost>
{% endhighlight %}

## 도메인 네임서버가 내 서버를 가리키게 설정
도메인만 있고 도메인에 걸린 IP가 없으면 앙꼬없는 찐빵이다. 네임서버는 도메인에 IP를 연결해주는 역할을 한다.<br />

도메인을 구입한 사이트에서 제공하는 네임서버를 사용할 경우 해당 사이트에 들어가 설정을 바꿀 수 있다.<br />
네임서버 설정에서 내 서버의 IP를 A레코드 설정에 저장하면 된다.
