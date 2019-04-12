---
layout: post
title: Dependency Injection 에 대해서
date: 2019-04-01 13:00:00 +0900
description: Remote Client # Add post description (optional)
img:  # Add image post (optional)
tags: [development]
categories : [OOP]
---

## Dependency Injection 이란?
 - 의존성 주입(DI)
 - 느슨한 결합을 가능하게 하는 디자인 패턴
 - 장점 : 
    - 클래스간 결합도 감소.
    - 코드 재사용성 증가.
    - 단위 테스트 용이.
    - 코드의 유지보수 용이.

ex ) 신발가게에서 신발을 주문하면 공장에서 만드는 시나리오
- Shoes.class(신발 정보를 나타내는 신발 클래스)
{% highlight csharp %}
public class Shoes
{
    public string Name{get; set;}
    public int Size{get; set;}
}
{% endhighlight %}

- IShoesFactory.interface(신발공장 인터페이스)
{% highlight csharp %}
interface IShoesFactory
{
    public Shoes MakeShoes();
}
{% endhighlight %}

- NShoes.class(인터페이스를 상속받은 NShoes 클래스)
{% highlight csharp %}
public class NShoes : IShoesFactory
{
    public Shoes MakeShoes()
    {
        return new Shoes()
        {
            Name = "df",
            Size = 250
        }
    }
}
{% endhighlight %}

- AShoes.class(인터페이스를 상속받은 AShoes 클래스)
{% highlight csharp %}
public class AShoes : IShoesFactory
{
    public Shoes MakeShoes()
    {
        return new Shoes()
        {
            Name = "adf",
            Size = 260
        }
    }
}
{% endhighlight %}

- ShoesStore.class(신발가게 클래스)
{% highlight csharp %}
public class ShoesStore
{
    IShoesFactory shoesFactory = null;
    public Shoes OrderShoes(string shoesName)
    {
        if("NShoes".Equals(shoesName))
        {
            shoesFactory = new NShoes();
        }
        ...

        return shoesFactory.MakeShoes();
    }
}
{% endhighlight %}

- Main
{% highlight csharp %}
static void Main(string[] args)
{
    ShoesStore shoesStore = new ShoesStore();
    Shoes myNesShoes = shoesStore.OrderShoes("NShoes");
}
{% endhighlight %}

## 위 코드 요약
 - ShoesStore class 와 NShoes, AShoes class 가 의존성을 맺고 있음.
 - 만약 새로운 공장 LShoes 에 주문하려면 ShoesStore 수정해야함.
 - 기존 공장 없어져도 ShoesStore 수정해야함.

## 의존성 제거하여 위 문제를 해결해보자.
- ShoesStore.class
{% highlight csharp %}
public class ShoesStore
{
    public Shoes OrderShoes(IShoesFactory shoesFactory)
    {
        return shoesFactory.MakeShoes();
    }
}
{% endhighlight %}

- Main
{% highlight csharp %}
static void Main(string[] args)
{
    ShoesStore shoesStore = new ShoesStore();
    Shoes myNesShoes = shoesStore.OrderShoes(new NShoes());
}
{% endhighlight %}

## 개선된 점
 - Shostore 는 MakeShoes 만 호출.
 - 신발 주문할 때 인터페이스를 통해 주문하여 의존성을 제거함.
 - Main 에서 요구에 맞춰 의존성(NShoes)를 주입한다.

참조 [https://hackersstudy.tistory.com/106?category=503176](https://hackersstudy.tistory.com/106?category=503176)