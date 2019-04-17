---
layout: post
title: condition_variable에 대해서
date: 2019-04-17 10:30:00 +0900
description: C++ Study # Add post description (optional)
img:  # Add image post (optional)
tags: [development, C++]
categories : [C++]
---

## condition_variable 언제 씀?
 - 스레드간 통신 원할 때
    - ex) 특정조건을 만족할 때까지 스레드를 블록하고 싶을 때.

## 사용방법
 - condition_variable:: wait, notify_one, notify_all 메서드를 이용
 - 다른 스레드가 일을 처리할 때까지 기다리게 할 수 있다.
 - wait 을 사용하기 위해서는,
 - 스레드가 먼저 mutex 를 점유해야 한다.
 - notify_... 함수를 호출하면 wait 하고 있는 조건변수에 알린다.
 - 그러면 해당 스레드가 깨어난다.

{% highlight cpp %}
condition_variable condition_value;
bool flag = false;
boost::mutex some_mutex;

void worker()
{
    {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        unique_lock<boost::mutex> lock(some_mutex);
        flag = true;
    }

    condition_value.notify_all();
}

int main()
{
    boost::thread th1(worker);
    unique_lock<boost::mutex> lock(some_mutex);
    if(!flag)
    {
        condition_value.wait(lock);
    }

    th1.join();
}

{% endhighlight %}