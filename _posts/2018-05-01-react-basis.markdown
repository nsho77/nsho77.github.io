---
layout: post
title: "[번역글] All the fundamental React.js concepts, jammed into this single Medium article"
date: 2018-05-01 10:40:00 +0900
description: medium 에서 올라온 글을 번역했다. 리액트에 관한 내용 # Add post description (optional)
img:  # Add image post (optional)
---

>medium 에 올라온 [글](https://medium.freecodecamp.org/all-the-fundamental-react-js-concepts-jammed-into-this-single-medium-article-c83f9b53eac2)을 번역해 보았다. 오역이 많을 거 같다.

지난 해 나는 100page 정도 되는 React.js 학습에 관한 책을 썼다. 올해 그 내용을 medium에 요약해보겠다.

이 글은 React가 무엇이고 (왜 써야 하는지)[https://medium.freecodecamp.org/yes-react-is-taking-over-front-end-development-the-question-is-why-40837af8ab76]에 대해 다루지 않을 것이다. 대신 Javascript와 DOM API 에 익숙한 사람을 대상으로 React.js의 기초에 대해 소개하는 글을 쓸 것이다.

아래의 모든 코드는 참고용이며 컨셉을 제공하기 위해 사용되었다. 대부분은 더 좋은 코드로 발전시킬 수 있을 것이다.

## Fundamental #1 : React 는 component 로 이루어져 있다.
React는 재사용 가능한 component를 사용하도록 만들어져 있다. 작은 component를 만들고 그것들을 이용해 더 큰 component를 만들 수 있다. 크기에 상관없이 모든 component는 재사용 가능하며 다른 프로젝트에서도 쓰일 수 있다.
가장 가단한 형태에서 react component는 plain-old Javascript 함수이다.

{% highlight javascript %}
function Button(props){
    //DOM Element를 return 한다.
    return <button type="submit">{props.label}</button>;
}
// 브라우저에 Button component를 랜더링한다.
ReactDOM.render(<Button label="Save" />,mountNode);
{% endhighlight %}

Button label의 중괄호는 나중에 설명할 거니까 지금은 생각할 필요없다. ReactDOM도 설명할 거다.
```ReactDOM.render```의 두번째 인자는 React가 컨트롤 하는 DOM Element 목적지다.
[jsComplete React Playground](https://jscomplete.com/react/) 에서 mountNode를 사용하면 ```ReactDOM.render()```가 실행된다.

Example 1에서 주목해야 할 내용:
* component의 이름은 대문자로 시작한다. HTML Elements와 React Elements를 함께 사용하기 때문에 지켜야 할 사항이다. 소문자로 이름을 지으면 HTML Elements로 인식한다. 예제에서 Button을 button으로 고치면 ReactDOM이 그냥 빈 HTML button을 만든다.

* 모든 component는 HTML elements처럼 속성인자를 받는데 React에서는 이를 props라고 부른다. 함수 component에서 props에 다른 이름을 붙일 수 있다.

* Button 함수 component에 HTML 같이 생긴 것을 return 했다. 이건 Javascript, HTML 이 아니고 React.js 도 아니다. 너무 많이 사용되어서 React 어플리케이션 default 처럼 느껴진다. 이를 JSX라고 부르며 Javascript 확장이다. 위의 function 리턴 을 다른 HTML로 바꿔도 모두 잘 동작한다.

##Fundamental #2: JSX가 도대체 무엇?

예제 1은 JSX를 대신해 오로지 React.js 로만 쓸 수 있다.

{% highlight javascript %}
// Example 2 - React component without JSX

function Button(props){
    return React.createElement(
        "button",
        {type:"submit"},
        props.label
    );
}

ReactDOM.render(
    React.createElement(
        Button,
        {label:"Save"}
    ), mountNode
);
{% endhighlight %}