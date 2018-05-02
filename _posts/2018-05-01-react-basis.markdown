---
layout: post
title: "[번역글] All the fundamental React.js concepts, jammed into this single Medium article"
date: 2018-05-01 10:40:00 +0900
description: medium 에서 올라온 글을 번역했다. 리액트에 관한 내용 # Add post description (optional)
img:  # Add image post (optional)
---

>medium 에 올라온 [글](https://medium.freecodecamp.org/all-the-fundamental-react-js-concepts-jammed-into-this-single-medium-article-c83f9b53eac2)을 번역해 보았다. 오역이 많을 거 같다.

지난 해 나는 100page 정도 되는 React.js 학습에 관한 책을 썼다. 올해 그 내용을 medium에 요약해보겠다.

이 글은 React가 무엇이고 [왜 써야 하는지](https://medium.freecodecamp.org/yes-react-is-taking-over-front-end-development-the-question-is-why-40837af8ab76)에 대해 다루지 않을 것이다. 대신 Javascript와 DOM API 에 익숙한 사람을 대상으로 React.js의 기초에 대해 소개하는 글을 쓸 것이다.

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

## Fundamental #2: JSX가 도대체 무엇?

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

```createElement``` 함수는 React 의 top-level API 이다. 배워야 할 API 가 8개가 있는데 이게 그중 하나이다. 이를 보면 React API가 얼마나 작은지 알 수 있다.

DOM 의 ```document.createElement``` 함수가 특정 tag name의 element를 생성하는 것처럼 React의 ```createElement``` 함수도 ```document.createElement```가 하는 일을 하면서 React component 를 만드는 일도 하는 higher-level 함수 이다. 우리는 예제 2에서 ```Button``` component를 이 함수를 이용해 만들어 보았다.

```document.createElement``` 와는 다르게 React는 많은 인자를 받을 수 있는데, 2번째 인자 이후 부터는 자식 element를 만드는데 사용된다. 따라서 ```createElement```는 tree를 만든다.

예를 봐보자.

{% highlight javascript %}
// Example 3 - React's createElement API

const InputForm = React.createElement(
    "form",
    { target:"_blank", action:"https://google.com/search" },
    React.createElement("div", null, "Enter input and click Search"),
    React.createElement("input", {name:"q", className:"input"}),
    React.createElement(Button, {label:"Search"})
);

// InputForm 이 Button component를 사용하고 있으므로 정의해줘야 한다.
function Button(props){
    return React.createElement(
        "button",
        {type: "submit"},
        props.label
    );
}

// 이제 InputForm을 render 에 바로 쓸 수 있다.
ReactDOM.render(InputForm, mountNode);
{% endhighlight%}

위 예에 대해서 생각해 봐야 할 점:
* InputForm 은 React component가 아니라 React element 다. 그래서 render에서 <InputForm> 이 아니라 바로 InputForm 을 쓴 것이다.

* ```React.createElement``` 함수는 세 번째 인자 부터 자식 element를 받는다.

* ```React.createElement``` 를 중첩하여 호출 할 수 있다. 왜냐하면 이게 다 JavaScript 이기 때문이다.

* ```React.createElement``` 의 2번째 인자는 속성이나 props 가 없는 경우 null 또는 빈 객체 일 수 있다. 

* React element 와 HTML element를 같이 쓸 수 있다.

* React API 는 DOM API 와 비슷하다. 그래서 DOM 의 class를 className 으로 쓴다. 솔직히 우리는 React API가 DOM API 가 됐음 좋겠다. 그게 더 좋으니까.

위의 코드는 브라우저가 React를 이해하는 방법이다. 브라우저는 JSX를 다룰 줄 모른다. 하지만 사람은 HTML 같은 코드를 더 편하게 사용한다. 이 때문에 JSX 합의가 존재하는 것이고 ```React.createElement``` 대신에 HTML과 비슷한 문법을 쓸 수 있다.

{%highlight javascript%}
// Example 4 - JSX (Example 3과 비교...)

const InputForm = 
    <form target="_blank" action="https://google.com/search">
        <div>Enter input and click Search</div>
        <input name="q" className="input" />
        <Button label="Search" />
    </form>;

function Button(props){
    return <button type="submit">{props.label}</button>;
}

ReactDOM.render(InputForm,mountNode)
{%endhighlight%}

몇 가지 주목해야 할 점:
* 이건 HTML 이 아니다. 우리는 ```class``` 대신에 ```className```을 쓰고 있다.
* tag의 끝에 semi-colon 을 보면 알 수 있듯이 HTML 문법 같지만 JavaScript 로 취급하고 있다.

이게 JSX 이다. 우리가 React component를 HTML 처럼 쓸 수 있게 만드는 합의다. 짱 좋다.

JSX 는 따로 쓰일 수 있다. React에서만 쓸 수 있는 거 아니다.

## Fundamental #3: JSX 어느 곳에서나 JavaScript expressions 을 사용할 수 있다.

중괄호를 이용하여, JSX 영역에서 언제든지 JavaScript Expression을 사용할 수 있다.

{%highlight javascript%}
// Example 5 - JSX 에서 JavaScript Expression 사용하기

const RandomValue = ()=>
    <div>{Math.floor(Math.random()*100)}</div>;

ReactDOM.render(<RandomValue />,mountNode);
{%endhighlight%}

JSX 에서 JavaScript 사용시 유일한 규칙은 오직 expression 만 된다는 것이다. 예를 들어 ```if``` 구문은 사용할 수 없지만 삼항연산자는 사용 할 수 있다.

JavaScript 변수 는 exrpession 이기 때문에 component가 props를 인자로 받으면 중괄호 안에 props를 사용할 수 있다. 우리는 Example1 의 ```Button``` component 에서 이렇게 사용했다.

JavaScript 객체도 expression 이다. 종종 중괄호 안에 객체를 사용하여 double 중괄호 처럼 보일 때가 있는데, 이건 그냥 중괄호 안에 있는 JavaScript 객체이다. 한 가지 예로 React에서 css 속성에 객체를 전달할때 를 들 수 있다.

{%highlight javascript%}
// Example 6 - React 에 css 객체 전달하기

const ErrorDisplay = ({message}) =>
    <div style ={ {color:'red', backgroundColor:'yellow'} }>{message}</div>;

ReactDOM.render(<ErrorDisplay 
    message="These aren't the droids you're looking for" 
    />, 
    mountNode);
{%endhighlight%}

인자 props 에서 message를 어떻게 비구조화 할당 했는지 주의깊게 봐라. 또, ```stlye``` 속성도 어떤 모습으로 생겼는지 살펴보자(다시 말하지만, DOM API 와 비슷하지만 HTML은 아니다). 우리는 style 속성에 값을 전달하기 위해 객체를 사용했다. JavaScript로 하는 것처럼 이 객체는 style을 정의한다(왜냐하면 우리가 JavaScript를 사용했으니까).

JSX 안에 React Element를 사용할 수 있다. 왜냐하면 이것또한 expression 이기 때문이다. React element는 함수 호출이어야 하는 점을 기억하자.

{% highlight javascript %}
// Example 7 - JSX 안에 React element를 사용해보자

const MaybeError = ({errorMessage}) =>
    <div>
        {errorMessage && <ErrorDisplay message={errorMessage}/>}
    </div>;

const ErrorDisplay = ({message}) =>
    <div style={ {color:'red', backgroundColor:'yellow'} }>
        {message}
    </div>;

ReactDOM.render(
    <MaybeError errorMessage={Math.random() > 0.5 ? 'Not good':''}
    />,
    mountNode
);
{% endhighlight %}