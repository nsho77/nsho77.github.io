---
layout: post
title: nInject 사용하는 간단한 방법
date: 2019-04-09 18:00:00 +0900
description: # Add post description (optional)
img: # Add image post (optional)
tags: [development]
categories : [Remote Client]
---

## 사용하는 이유는?
 - 의존성을 주입할 수 있음.
 - 주입 할 수 있기 때문에 의존성을 제거할 수 있음.
 
## 사용하는 방법은?
 - manual dependency injection 을 Ninject로 바꿔보자.

### 코드 설명
 - formHandler 가 mailSender 를 가지고 있음.
 - mailSender 를 유연하게 쓰고 싶음.
{% highlight csharp %}
class Program
{
    static void Main(string[] args)
    {
        IMailSender mailSender = new MockMailSender();
        FormHandler formHandler = new FromHandler(mailSender);
        fromHandler.Handle("test@test.com");
    }
}
{% endhighlight %}

위의 manual dependecy injection 을 Ninject 로 바꾸면

{% highlight csharp %}
using Ninject;

class Program
{
    static void Main(string[] args)
    {
        var kernel = new StandardKernel();
        Bind<IMailSender>.To<MockMailSender>();
        FormHandler formHandler = new FromHandler(kernel.Get<IMailSender>());
        fromHandler.Handle("test@test.com");
    }
}
{% endhighlight %}