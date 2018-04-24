---
layout: post
title: express 의 router 함수의 기능은??
date: 2018-04-24 11:38:00 +0900
description: express의 router 함수의 기능은 무엇일까. # Add post description (optional)
img: how-to-start.jpg # Add image post (optional)
tags: [Programming, Learn, Node] # add tag
---
>아래의 소스는 [Velopert](https://velopert.com/1921)님의 블로그에 있는 소스입니다.

```javascript
import express from 'express';

const router = express.Router();

router.post('/signup', (req, res)=>{
    /* to be implemented */
    res.json({success: true});
});

router.post('/signin',(req, res)=>{
    /* to be implemented */
    res.json({success : true});
})

router.get('/getinfo', (req, res)=>{
    res.json({info: null});
});

router.post('/logout', (req, res)=>{
    return res.json({success: true});
});

export default router;
```

>express 는 Node.js 에서 사용할 수 있는 웹 어플리케이션 프레임워크이다. <br/>여기서 제공하는 여러 기능 중 Router() 에 대해서 알아보겠다.

velopert 님의 블로그에 있는 튜토리얼을 진행하다 위와 같은 소스를 만났다.

이것 말고도 모르는 것 투성이지만 오늘은 Router() 의 기능에 대해서 알아보자

```javascript
import express from 'express';

const router = express.Router();
```
express 모듈을 불러오고 새로운 Router 를 만들었다. 이 Router 의 이름은 router!

## Route 란??

client 의 요청을 위한 URL 스키마. 서버와 클라이언트의 통신을 위한 인터페이스를 제공해주는 역할을 한다.<br/>
ex) Route에 GET 요청을 했다면 해당 URL 에 서버 자원을 요구한다는 의미로 해석한다.<br/>
따라서 Router는 특정 URL 에 어떤 기능을 하게 만들건지 지정하는 애플리케이션 이라고 이해하면 될 거 같다.

```javascript
router.post('/signup', (req, res)=>{
    /* to be implemented */
    res.json({success: true});
});

router.post('/signin',(req, res)=>{
    /* to be implemented */
    res.json({success : true});
})

router.get('/getinfo', (req, res)=>{
    res.json({info: null});
});

router.post('/logout', (req, res)=>{
    return res.json({success: true});
});
```
첫번째 router가 하는 일은<br/>
클라이언트로 부터 /signup 의 url 에 post 요청을 받으면 콜백함수를 실행하는 것이다.

## 파라미터 req, res 가 의미하는 것은?
클라이언트가 서버에 요청을 전달하라는 명령을 내리면 브라우저에서 클라이언트의 요청을 문서에 담아 서버에 전달한다.<br />
클라이언트의 요청이 담긴 문서를 request 객체라고 한다.<br />
response 객체는 위와 방향이 반대이다.<br/>서버에서 클라이언트의 요청을 처리한 결과를 문서에 담아 보내는데 이 문서를 response 객체라고 한다.<br />
express 는 req, res 객체가 메소드를 갖게 만들었는데 res.json()은 response 를 json 타입으로 보내는 기능을 한다. (json() 함수에 json 인자를 전달하여 사용한다.)

## router의 메소드 use 가 하는 일은 무엇?
```javascript
import express from 'express';
import account from './account';

const router = express.Router();
router.use('/account', account);

export default router;
```
use 메소드는 express 에서 middleware 를 등록해주는 역할을 한다.<br/>
middleware 가 동작하는 방법은 middleware 로 등록한 로직을 실행하고 next() 를 호출하면 다음에 등록된 middleware가 실행되는 방식이다.<br/>
아래의 코드를 보자.(출처 : http://webframeworks.kr/getstarted/expressjs/)
```javascript
var express = require('express');
var app = express();

app.use(function middleware1(req, res, next) {
  console.log('middleware1: 인증작업...')
  if (/*인증 되었으면*/) {
    next();
  } else {
    next(new Error('Unauthorized'))
  }
});
app.use(function middleware2(req, res, next) {
  console.log('middleware2: 로깅작업...')
  next();
});
app.use(function middleware3(err, req, res, next) {
  if (err) {
    if (err.message === 'Unauthorized') {
      return res.status(401).send('Unauthorized');
    }
    return res.satus(400).send('BadRequest');
  }
  next()
});
```
위의 코드 진행 순서는 <br />
첫 번째 middleware ->(에러가 없으면) 두 번째 middleware -> ...<br />
첫 번째 middleware ->(에러가 있으면) 세 번째 middleware -> ...<br />
*next 인자로 error 객체를 넣으면 err 을 파라미터로 가지고 있는 middleware 가 호출된다.<br/>

middleware 등록방법은 <br />
use 메소드의 첫번째 인자로 path를 전달하고 두번째 인자로 middleware function 을 전달하면 된다.<br />
request 의 path가 첫번째 인자와 일치하면 middleware function 을 실행한다. <br />
아래 코드 출처 : http://expressjs.com/ko/4x/api.html#router.use
```javascript
var express = require('express');
var app = express();
var router = express.Router();

// simple logger for this router's requests
// all requests to this router will first hit this middleware
router.use(function(req, res, next) {
  console.log('%s %s %s', req.method, req.url, req.path);
  next();
});

// this will only be invoked if the path starts with /bar from the mount point
router.use('/bar', function(req, res, next) {
  // ... maybe some additional /bar logging ...
  next();
});

// always invoked. path default 는 / 이다. 따라서 항상 호출된다.
router.use(function(req, res, next) {
  res.send('Hello World');
});

/// router 객체를 middleware로 등록 할 수 있다.
app.use('/foo', router);

app.listen(3000);
```
