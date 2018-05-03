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
void pivotSort(int* arr, int left, int right)
{
    // 피벗 요소는 정렬 해야 할 배열의 첫 번째 요소로 정한다.
    int pivot = arr[left];
    
    // 피벗 앞과 배열 맨끝을 가리키는 포인터를 정한다.
    int low = left + 1;
    int high = right;

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
}
{% endhighlight %}

![I and My friends]({{site.baseurl}}/assets/img/we-in-rest.jpg)

Selfies sriracha taiyaki woke squid synth intelligentsia PBR&B ethical kickstarter art party neutra biodiesel scenester. Health goth kogi VHS fashion axe glossier disrupt, vegan quinoa. Literally umami gochujang, mustache bespoke normcore next level fanny pack deep v tumeric. Shaman vegan affogato chambray. Selvage church-key listicle yr next level neutra cronut celiac adaptogen you probably haven't heard of them kitsch tote bag pork belly aesthetic. Succulents wolf stumptown art party poutine. Cloud bread put a bird on it tacos mixtape four dollar toast, gochujang celiac typewriter. Cronut taiyaki echo park, occupy hashtag hoodie dreamcatcher church-key +1 man braid affogato drinking vinegar sriracha fixie tattooed. Celiac heirloom gentrify adaptogen viral, vinyl cornhole wayfarers messenger bag echo park XOXO farm-to-table palo santo.

>Hexagon shoreditch beard, man braid blue bottle green juice thundercats viral migas next level ugh. Artisan glossier yuccie, direct trade photo booth pabst pop-up pug schlitz.

Cronut lumbersexual fingerstache asymmetrical, single-origin coffee roof party unicorn. Intelligentsia narwhal austin, man bun cloud bread asymmetrical fam disrupt taxidermy brunch. Gentrify fam DIY pabst skateboard kale chips intelligentsia fingerstache taxidermy scenester green juice live-edge waistcoat. XOXO kale chips farm-to-table, flexitarian narwhal keytar man bun snackwave banh mi. Semiotics pickled taiyaki cliche cold-pressed. Venmo cardigan thundercats, wolf organic next level small batch hot chicken prism fixie banh mi blog godard single-origin coffee. Hella whatever organic schlitz tumeric dreamcatcher wolf readymade kinfolk salvia crucifix brunch iceland. Literally meditation four loko trust fund. Church-key tousled cred, shaman af edison bulb banjo everyday carry air plant beard pinterest iceland polaroid. Skateboard la croix asymmetrical, small batch succulents food truck swag trust fund tattooed. Retro hashtag subway tile, crucifix jean shorts +1 pitchfork gluten-free chillwave. Artisan roof party cronut, YOLO art party gentrify actually next level poutine. Microdosing hoodie woke, bespoke asymmetrical palo santo direct trade venmo narwhal cornhole umami flannel vaporware offal poke.

* Hexagon shoreditch beard
* Intelligentsia narwhal austin
* Literally meditation four
* Microdosing hoodie woke

Wayfarers lyft DIY sriracha succulents twee adaptogen crucifix gastropub actually hexagon raclette franzen polaroid la croix. Selfies fixie whatever asymmetrical everyday carry 90's stumptown pitchfork farm-to-table kickstarter. Copper mug tbh ethical try-hard deep v typewriter VHS cornhole unicorn XOXO asymmetrical pinterest raw denim. Skateboard small batch man bun polaroid neutra. Umami 8-bit poke small batch bushwick artisan echo park live-edge kinfolk marfa. Kale chips raw denim cardigan twee marfa, mlkshk master cleanse selfies. Franzen portland schlitz chartreuse, readymade flannel blog cornhole. Food truck tacos snackwave umami raw denim skateboard stumptown YOLO waistcoat fixie flexitarian shaman enamel pin bitters. Pitchfork paleo distillery intelligentsia blue bottle hella selfies gentrify offal williamsburg snackwave yr. Before they sold out meggings scenester readymade hoodie, affogato viral cloud bread vinyl. Thundercats man bun sriracha, neutra swag knausgaard jean shorts. Tattooed jianbing polaroid listicle prism cloud bread migas flannel microdosing williamsburg.

Echo park try-hard irony tbh vegan pok pok. Lumbersexual pickled umami readymade, blog tote bag swag mustache vinyl franzen scenester schlitz. Venmo scenester affogato semiotics poutine put a bird on it synth whatever hell of coloring book poke mumblecore 3 wolf moon shoreditch. Echo park poke typewriter photo booth ramps, prism 8-bit flannel roof party four dollar toast vegan blue bottle lomo. Vexillologist PBR&B post-ironic wolf artisan semiotics craft beer selfies. Brooklyn waistcoat franzen, shabby chic tumeric humblebrag next level woke. Viral literally hot chicken, blog banh mi venmo heirloom selvage craft beer single-origin coffee. Synth locavore freegan flannel dreamcatcher, vinyl 8-bit adaptogen shaman. Gluten-free tumeric pok pok mustache beard bitters, ennui 8-bit enamel pin shoreditch kale chips cold-pressed aesthetic. Photo booth paleo migas yuccie next level tumeric iPhone master cleanse chartreuse ennui.
