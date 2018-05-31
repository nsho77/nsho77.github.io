---
layout: post
title: "[번역글] All the fundamental React.js concepts, jammed into this single Medium article"
date: 2018-05-01 10:40:00 +0900
description: medium 에서 올라온 글을 번역했다. 리액트에 관한 내용 # Add post description (optional)
img:  # Add image post (optional)
tags: [development, translate]
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

{% highlight js %}
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

```MaybeError``` component 는 ```errorMessage```에 string이 전달 되면 보일 것이다. React는 ```{true}```, ```{false}```, ```{undefined}```, ```{null}```을 element 자식이라고 판단하며 브라우저에 아무것도 그리지 않는다.

JSX 안에서 collection 에 대한 JavaScript method 전부를 사용할 수 있다(```map```,```reduce```,```filter```,```concat``` 등). 다시 말하지만, 이들 모두는 expression을 return 하기 때문에 가능한 것이다.

{% highlight js%}
// Example 8 JSX 안에 array map 사용하기

const Doubler = ({value=[1,2,3]})=>
    <div>
        {value.map(e=>e*2)}
    </div>;

ReactDOM.render(<Doubler />,mountNode);
{% endhighlight %}

이번엔 prop 에 default 값을 줘 보았다. 이게 다 JavaScript 라는 것을 명심하자. ```div```안에 array expression 을 넣었다. React에서는 이를 text node에 저장하여 보여준다.

## Fundamental #4: JavaScript Class 와 React component를 함께 사용할 수 있다.

간단한 함수 component는 심플한 요청을 처리하기엔 좋지만, 우리는 때때로 더 많은 것을 해야 할 필요가 있다. React는 JavaScript class 문법도 지원한다. 여기 class 문법으로 만든 ```Button``` component(Example 1)를 보자.


{% highlight js%}
// Example 9 - JavaScript 클래스로 component 만들기

class Button extends React.Component{
    render(){
        return <button>{this.props.label}</button>
    }
}

ReactDOM.render(<Button label="Save"/>, mountNode);
{% endhighlight %}

클래스 문법은 간단하다. ```React.Component```(우리가 배워야 하는 top-level API 중 하나) 를 상속하는 클래스를 하나 만든다. 이 클래스는 하나의 instance 함수 ```render()```를 정의하고 이 render 함수는 가상의 DOM element를 return 한다. ```Button``` 클래스 기반 component를 사용할 때마다(예를 들면 ```<Button ... />``` 이렇게.. ) React는 클래스 기반 component를 사용하여 객체를 만들고 만든 객체를 DOM tree 에 DOM 으로 렌더링한다.

이것 때문에 JSX 안에서 ```this.props.label```을 사용하는 것이다. 클래스 component를 통해 구체화 된 모든 element들은 구체화된 특별한 속성 ```props```를 가지고 있는데 이 객체가 만들어질때 전달된 모든 값이 여기에 들어있다.

component의 단일한 사용을 위한 구체화된 class를 이용하여, 우리 마음대로 커스터마이징 하여 class를 구체화 시킬 수 있다. 예를 들어, 인스턴스가 만들어진 뒤에 JavaScript 의 ```construct``` 함수를 이용하여 커스터 마이징을 할 수 있다.

{% highlight js%}
// Example 10 - component instance 를 커스터마이징 하기.

class Button extends React.Component{
    constructor(props){
        super(props);
        this.id = Date.now();
    }
    render(){
        return <button id={this.id}>{this.props.label}</button>;
    }
}

ReactDOM.render(<Button label="Save" />, mountNode);
{% endhighlight %}

클래스 함수를 정의하고 원하는 곳 어디에서나 사용가능하다. 함수가 JSX 를 return 하는 것도 가능하다.

{%highlight js%}
//Example 11 - 클래스 property 사용하기

class Button extends React.Component{
    clickCounter = 0;

    handleClick = ()=>{
        console.log(`Clicked: ${++this.clickCounter}`);
    }
    
    render(){
        return(
            <button id={this.id} onClick={this.handleClick}>
                {this.props.label}
            </button>
        );
    }
}

ReactDOM.render(<Button label="Save" />, mountNode);
{%endhighlight%}

예제 11 에서 눈여겨 보아야 할 점:
* ```handleClick``` 함수는 새롭게 제안된 [class-field syntax](https://github.com/tc39/proposal-class-fields)로 작성했다. 이건 아직 stage-2 옵션이지만 여러 이유로 인스턴스화 된 component 에 접근하는 데 가장 좋은 옵션이다(arrow function 덕분에). 그러나 이를 사용하려면 stage-2 를 compile 할 수 있는 Babel 과 같은 compiler를 사용해야 한다.

* ```clickCounter``` 도 같은 class-field 문법을 사용했다. 이 문법을 사용하면 class constructor 를 호출 하지 않아도 된다.

* React의 ```onClick``` 속성으로 ```handleClick``` 함수를 넘길 때 이 함수를 호출하지 않고 함수 reference를 넘겨주었다. 함수를 장착할 때 함수를 호출하는 실수를 많이들 한다.

{% highlight js%}
// Wrong:
onClick = {this.handleClick()}

//Right:
onClick = {this.handleClick}
{% endhighlight %}

## Fundamental #5: React Event: 두 가지 중요한 차이점

DOM API 와 다르게 React Event 를 다룰 때 다른 점 두 가지가 있다.

* React component 속성은(event 포함) camelCase 를 사용한다. ```onClick``` 이렇게 ```onclick``` 이거 말고.

* event handler 를 전달할 때 string 이 아닌 객체 reference를 전달한다. ```onClick={handleClick}``` 이렇게 ```onClick="handleClick"``` 이거 말고.

React 는 event handling 최적화를 위해 DOM event object를 감싸는 object를 갖고 있다. 그러나 event handler 안에서는 DOM event object 와 같은 메서드를 사용할 수 있다. React는 DOM event object를 감싼 object를 모든 handle 호출에 전달한다. 예를 들어서 default submit action을 차단하고 싶을 때 아래와 같이 할 수 있다.

{%highlight js%}
// Example 12 - event 객체로 작업해보기
class Form extends React.Component{
    handleSubmit = (event)=>{
        event.preventDefault();
        console.log('Form submitted');
    };

    render(){
        return(
            <form onSubmit={this.handleSubmit}>
                <button type="submit">Submit</button>
            </form>
        );
    }
}

ReactDOM.render(<Form />,mountNode);
{%endhighlight%}

## Fundamental #6: 모든 component 는 story가 있다.

이어지는 이야기는 class component(```React.Componet```를 상속한) 에 해당된다. 함수 component는 조금 다르다.

1. 첫째, component 로 element를 만드는 React template을 만든다.

2. React 어딘가에서 쓸 수 있게 component를 만든다. 예를 들어 다른 component render 함수 안이나 ```ReactDOM.render``` 에서..

3. React는 element를 instance 로 만들고 여러 개의 props 를 전달하는데 props는 ```this.props```로 접근 가능하다. step 2에서 전달한 것과 props 는 같은 것이다.

4. JavaScript 이기 때문에 constructor 를 정의했다면 호출된다. 이게 우리가 처음으로 호출한 것이다. : component lifecycle methods.

5. React는 render method의 결과를 계산한다.(가장의 DOM node를 이용함)

6. React가 element를 randering 하는 게 처음이라면 React는 browser와 통신할 것이다(DOM API를 이용해서). 그리고 element를 보여줄 것이다. 이 과정을 보통 mounting이라고 부른다.

7. 그다음에 React는 ```componentDidMount``` 라는 lifecycle method를 호출한다. 브라우저 상에 실제 있는 DOM 에 작업을 하고 싶을 때 이 메소드를 사용할 수 있다. 이 lifecycle method 전에는 우리가 작업하는 DOM 은 모두 가상의 DOM 이다.

8. 일부 component의 이야기는 여기서 끝난다. 일부 component는 브라우저의 DOM에서 여러 이유로 unmount 된다. 이 과정이 진행되기 전에 React는 ```componentWillUnmount```라는 lifecycle method를 호출한다.

9. mount 된 element의 상태는 언제든지 변할 수 있다. 부모 element가 다시 rendering 될 수도 있다. 이런 경우에, mount 된 element 는 다른 props를 받을 수 있다. React의 마법은 여기서 시작된다. 이전까지 에서는 사실 React가 필요하지 않다.

component의 이야기는 계속된다. 그러나 그전에 위에서 말한 상태, state에 대해서 알아야 한다.

## Fundamental #7: React component 는 private state를 가질 수 있다.

이어지는 내용도 class component에만 해당된다. 어떤 이들은 보여주기만 하는 component를 멍청하다고 한다는 걸 얘기했었나?

```state``` 속성은 특별하다. React는 모든 component의 state 변화를 모니터한다. React가 이를 효과적으로 수행하게 하기 위해서 우리는 state 변화를 React API를 이용하여 만들어야 한다. ```this.setState``` 가 그것이다.

{%highlight js%}
// Example 13 - setState API

class CounterButton extends React.Component{
    state = {
        clickCounter : 0,
        currentTimestamp: new Date(),
    };

    handleClick = ()=>{
        this.setState((prevState)=>{
            return { clickCounter: prevState.clickCounter + 1};
        });
    }

    componentDidMount(){
        setInterval(()=>{
            this.setState({currentTimestamp: new Date()})
        },1000);
    }

    render(){
        return(
            <div>
                <button onClick={this.handleClick}>Click</button>
                <p>Clicked: {this.state.clickCounter}</p>
                <p>Time:{this.state.currentTimestamp.toLocaleString()}</p>
            </div>
        )
    }
}

ReactDOM.render(<CounterButton />, mountNode);
{%endhighlight%}

이 예제는 반드시 이해해야 한다. 위 예제는 React가 어떻게 움직이는지를 잘 보여준다. 이후 배워야할 게 몇 개 있지만 JavaScript 스킬에 지나지 않는다.

예제 13을 하나씩 분석해 보자. class fields 부터 시작한다. class field는 2개 있다. ```state``` field 는 object 인데 ```clickCounter``` key를 가지고 있고 value는 0이다. ```currentTimestamp``` key는 new Date() value로 시작한다.

두 번째 class field 는 ```handleClick``` 함수이다. render메소드 안에 button element의 event 로 전달된다. ```handleClick``` 메소드는 ```setState``` 메소드를 이용해서 component instance state를 변경한다.

render 메소드 에서 일반적인 문법으로 state의 두 속성을 사용한다. 여기엔 특별한 API가 없다.

이제 두 가지 다른 방법으로 state update 하는 과정을 살펴보자.

1. object를 return 하는 함수를 전달한다. ```handleClick``` 함수에서 사용한 방법

2. 일반적인 object를 전달하는 방법. interval 함수의 callback 에서 사용.

두 가지 방법 모두 가능하다. 그러나 첫 번째 방법을 선호한다. interval 에서 callback 으로 사용하면 state를 쓰기만 가능하고 읽기는 불가능하다. 의심스럽다면 인자로 함수를 넣어서 사용해 보라. ```setState``` 는 비동기적인 방법으로 처리되어야 하기에 race condition이 더 안전하다.

state는 어떻게 update 해야 할까? update 원하는 새로운 값을 가진 object를 return 해야 한다. 두 가지 경우에 ```setState```를 어떻게 호출했는지 보면, 한 가지 속성만 전달하고 두 가지 속성은 전달하지 않았다. ```setState``` 는 현재의 state와  merge 되기 때문에 이렇게 사용하여도 괜찮다. 그러니까 ```setState```를 통해 return 하지 않은 속성은 변하지 않게 내버려 둔다는 뜻이다(그러나 삭제하는 것도 아니다).

## Fundamenta #8: React 는 반응한다.

React 의 이름은 state 의 변화에 react 하기 때문에 지어진 것이다(반응은 아니고 스케줄에 따라). React 이름이 Schedule 이었어야 한다는 농담도 있다.

그러나 사람이 보기엔 어떤 component state의 변화가 생기면 React는 그에 반응하여 update하고 자동으로 browser DOM에 적용하는(만약 필요하다면) 것 같이 보인다.

render 함수의 input 대해서 생각해보자.

* 부모가 props를 전해준다.
* 내부 private state 는 언제든 update 될 수 있다.

render 함수의 input이 변하게 되면 output 도 변하게 될 것이다.

React 는 render 의 역사를 기록한다. 이전 render와 다른 render가 생기면 React는 이것과 그것 사이의 차이를 계산하여 효과적으로 실제 DOM operation에 전달하고 DOM에서 실행된다.

## Fundamental #9: React는 당신의 agent 다.

React는 browser와 통신하기 위해 우리가 고용한 agent 다. ...(중략)... 우리는 Mr.Browser와 말하기 싫어한다. 그리고 React는 우리를 위해 이런 궂은 일들을 해준다.

## Fundamental #10: 모든 React component 는 이야기가 있다.(part2)

component의 state가 변할 때 어떤 마법이 벌어지는 지 아니까 몇 가지 process concepts 에 대해 알아보자.

1. component는 state가 update 되거나 부모가 전달하는 props가 변경되면 re-render 되기도한다.

2. 부모가 전달하는 props가 변경되면 React는 ```componentWillReceiveProps``` lifecycle method를 호출한다.

3. state object가 변경되거나 전달받은 props가 변경되면 React는 중요한 결정을 해야 한다. DOM 에 있는 component를 update 해야 하나? 이런 이유로 ```shouldComponentUpdate``` 라는 중요한 lifecycle method가 호출된다. 이 메소드는 진짜 질문이다. 그리니까 렌더링 process를 커스터 마이징하거나 최적화 시킬때 true or false 를 반환하여 질문에 답해야 한다.

4. custom ```shouldComponentUpdate``` 가 없다면 React는 대부분의 상황에 알맞게 작동하게끔 영리하게 동작한다.

5. 첫 번재로, React는 ```componentWillUpdate``` lifecycle method를 호출한다. React는 새로운 render output과 마지막 render output 을 비교하여 차이를 계산한다.

6. output이 정확히 같다면 React는 아무 일도 하지 않는다(Mr.Browser와 이야기할 필요없다).

7. 차이가 있다면 React는 우리가 전에 본 것처럼 차이를 browser에 전달한다.

8. update process가 진행되면(output이 똑같더라도), React는 ```componentDidUpdate```라는 마지막 lifecycle method를 호출한다.

Lifecycle method는 해결하기 곤란한 여러 문제들에 해법으로 사용할 수 있다. 당신이 이와 관련하여 별 일을 하지 않으면 Lifecycle method 없는 application을 만들 수도 있다. Lifecycle method 는 application에서 무슨 일이 일어나고 있는지 분석 할때, React의 성능을 최적화 시킬때 유용한 도구로 사용할 수 있다.

이게 다다. 믿던지 말던지 위의 내용이면 당신은 React application을 개발할 수 있다. 더 배우고 싶다면 [Learn React.js by Building Games](https://www.amazon.com/dp/1986393895/ref=as_li_ss_tl?ie=UTF8&linkCode=sl1&tag=agil05e-20&linkId=dbcbdb04a40ba4095180f430eb5e0113) 책을 확인해보라.

React 또는 Node를 배우고 싶으면 나의 책을 확인해보라 :
* [Learn React.js by Building Games](https://www.amazon.com/dp/1986393895/ref=as_li_ss_tl?ie=UTF8&linkCode=sl1&tag=agil05e-20&linkId=dbcbdb04a40ba4095180f430eb5e0113)
* [Node.js Beyond the Basics](https://www.amazon.com/dp/1986394115/ref=as_li_ss_tl?ie=UTF8&linkCode=sl1&tag=agil05e-20&linkId=e9289a6988710695ec7313f813a7a1fd)