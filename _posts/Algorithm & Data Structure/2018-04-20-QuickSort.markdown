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
3. 피벗 요소와 우선순위가 가장 높은 요소의 자리를 바꾼다.

이를 재귀적으로 구현한다.

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
    // pivot이 1이면 low는 6+이고 high 는 0이 된다.
    // pivot과 high를 바꾸니 바뀌지 않는다..
    // 만약 low 와 high 가 같다면 어떻게 될까?
    // 똑같다 low 와 high가 같아질 때까지 반복하는 이유는 마지막 요소까지 모두 정렬해야 하기 때문이다.

    // 문제1 : 바깥 while을 반복할 때 low, high 가 움직이지 않으며 이후 연산을 반복할 수 있다.
    // (while 문 안의 if 덕분에 이런 경우가 생기지 않는다. low 와 high를 교환했다면 low가 늘어나고 high는 줄어들기 때문이다.)
    // 문제2 : 역순으로 정렬됐을 때 low 가 무한정 늘어날 수 있다.( debug 해본 결과 배열 크기보다 늘어나지만 무한정 늘어나지 않는다.)
    // 문제3 : low가 high 보다 커지면 pivot을 어디나 넣을 것인가. 
    // ( 이 문제제기는 잘 못 되었다. pivot을 low와 high 사이에 넣는 게 아니라 pivot과 high를 교환하는 것이다.)
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
   ...
}
{% endhighlight %}

## 3. 피벗 요소와 우선순위가 가장 높은 요소의 자리를 바꾼다.
{% highlight cpp %}
int pivotSort(int* arr, int left, int right)
{
    int pivot = arr[left];
    int low = left + 1;
    int high = right;

    while(left <= right)
    {
        while(pivot > arr[low])
            low++;
        while(pivot < arr[high])
            high--;
        
        if(low<=high)
        {
            int temp = arr[low];
            arr[low] = arr[high];
            arr[high] = temp;
        }
    }
    // 피벗 요소와 우선순위가 가장 높은 요소의 자리를 바꾼다.
    int temp = arr[left];
    arr[left] = arr[high];
    arr[high] = temp;

    return high; 
}
// pivotSort 함수를 호출한다.
void pivotSortCall(int* arr, int left, int right)
{
    // 재귀적으로 호출한다
    // 먼저 pivot을 하고 나누고 pivot 하는 과정을 거친다.

    if(left > right)
        return;

    int pivoted = pivotSort(arr, left, right);
    pivotSortCall(arr, left, pivoted-1);
    pivotSortCall(arr, pivoted+1, right);
}

{% endhighlight %}

