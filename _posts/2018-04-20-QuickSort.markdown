---
layout: post
title: Quick 정렬
date: 2018-04-20 19:10:20 +0900
description: Quick Sort를 직접 구현해보면서 공부하자 # Add post description (optional)
img: i-rest.jpg # Add image post (optional)
tags: [Algorithm, Programming]
---
Quick Sort는 정렬하기 위한 배열에서 pivot 요소를 택한 다음 pivot 요소 왼쪽에는 우선순위가 높은 요소, 오른쪽에는 우선순위가 요소로 정렬하는 알고리즘이다. 알고리즘 성능이 O(logN)으로 아주 효과적이다. 구현할 순서는 다음과 같다.

1. 피벗 요소를 선택하고 피벗 앞 요소와 배열의 맨 끝 요소를 가리키는 포인터를 정의한다.
2. 두 포인터가 만날때까지 정렬한다.
3. 피벗 요소를 정렬된 요소 사이로 옮긴다.

## 1. 피벗 요소를 선택하고 배열을 양쪽을 가리키는 포인터를 정의한다.
{% highlight cpp %}
// pivotSort 라고 함수 이름을 정했고 매개변수는 배열과 정수형 데이터 두 개를 받는다.
// 두 개의 요소는 정렬해야 할 배열 시작과 끝을 가리킨다.
void pivotSort(int* arr, int left, int right)
{
    // 피벗 요소는 정렬 해야 할 배열의 첫 번째 요소로 정한다.
    int pivot = arr[left];
    
    // 피벗 앞과 배열 맨끝을 가리키는 포인터를 정한다.
    int low = left + 1;
    int high = right;
}
{% endhighlight %}

## 2. 두 포인터가 만날 때까지 정렬한다.
{% highlight cpp %}
int pivotSort(int* arr, int left, int right)
{
    // 피벗 요소는 정렬 해야 할 배열의 첫 번째 요소로 정한다.
    int pivot = arr[left];
    
    // 피벗 앞과 배열 맨끝을 가리키는 포인터를 정한다.
    int low = left + 1;
    int high = right;

    // 배열을 가리키는 포인터 두 개가 만날 때까지 정렬을 반복한다.
    // 왼쪽에서는 pivot 보다 우선순위가 낮은 데이터를 찾고
    // 오른쪽에서는 pivot 보다 우선순위가 높은 데이터를 찾는다.
    // 만약 1,2,3,4,5,6,7 이면...
    // pivot이 1이고 low는 1이고 high 는 0이 된다.
    // 만약 low 와 high 가 같다면 어떻게 될까?
    // 똑같다 low 와 high가 같아질 때까지 반복하는 이유는 마지막 요소까지 모두 정렬해야 하기 때문이다.

    // 문제1 : 바깥 while을 반복할 때 low, high 가 움직이지 않으며 이후 연산을 반복할 수 있다.
    // 문제2 : 역순으로 정렬됐을 때 low 가 무한정 늘어날 수 있다.
    // 문제3 : low가 high 보다 커지면 pivot을 어디나 넣을 것인가.
    while(low <= high)
    {
        while(pivot > arr[low])
            low++;

        while(pivot < arr[high])
            high--;
        
        if(low <= high)
        {
            int temp = arr[low];
            arr[low] = arr[high];
            arr[high] = temp;
        }
    }
    int temp = arr[left];
    arr[left] = arr[high];
    arr[high] = temp;

    return high; 
}

// pivotSort 함수를 호출한다.
void pivotSortCall(int* arr, int left, int right)
{
    if(left > right)
        return;

    int pivoted = pivotSort(arr, left, right);
    pivotSortCall(arr, left, pivoted-1);
    pivotSortCall(arr, pivoted+1, right);
}

{% endhighlight %}

![I and My friends]({{site.baseurl}}/assets/img/we-in-rest.jpg)

