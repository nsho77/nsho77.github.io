---
layout: post
title: Git 브랜치와 Merge 기초
date: 2018-04-27 15:45:00 +0900
description: Git 의 공식 설명서를 읽고 정리해 보았다. # Add post description (optional)
img: workflow.jpg # Add image post (optional)
tags: [Programming, Git] # add tag
---
깃의 [다음 페이지](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) 에서 상세한 설명을 볼 수 있습니다.<br />
velopert 님의 튜토리얼 중 git checkout 하는 부분이 있어서 관련 내용을 찾아보던 중 정리한 내용이다.  

###  브랜치란?
head가 가리키는 작업영역이다. 기본적으로 master 브랜치에서 작업하고 이 branch 와 다른 작업을 하고 싶을 때 새로운 branch를 생성해 작업 할 수 있다.<br />
작업하는 branch를 바꾸면 head 가 해당 branch를 가리킨다.<br />
다음은 branch를 생성하는 명령어이다. branch 명은 iss53으로 하자
```bash
$ git branch iss53
```
다음은 iss53으로 head 가 가리키는 영역을 바꾸는 명령이다.
```bash
$ git checkout iss53
```
branch iss53에서 commit을 하면 다음 그림과 같은 상태가 된다.
<img src="https://git-scm.com/book/en/v2/images/basic-branching-3.png" alt="make branch">

그런데 master 브랜치에서 해결해야 할 이슈가 생겼다면 어떻게 해야 할까? 일단 master 브랜치로 checkout 해야 한다.
이슈를 해결하고 commit 하기 위해서 새로운 브랜치 hotfix를 만들고 commit 한다. 그러면 아래와 같은 상태가 된다. <br />
<img src="https://git-scm.com/book/en/v2/images/basic-branching-4.png" alt="Hotfix branch based on `master`.">

master 브랜치로 와서 hotfix 브랜치의 내용을 merge 하여 수정 사항을 반영한다. hotfix는 master 브랜치를 base로 하기 때문에
merge 를 할 경우 단순히 master 가 가리키는 커밋이 hotfix가 가리키는 커밋으로 바뀌게 된다. 이를 master 브랜치가 hotfix로 fastforward 되었다고 한다.<br />
<img src="https://git-scm.com/book/en/v2/images/basic-branching-5.png" alt="`master` is fast-forwarded to `hotfix`.">
![Macbook]({{site.baseurl}}/assets/img/mac.jpg)
Man bun umami keytar 90's lomo drinking vinegar synth everyday carry +1 bitters kinfolk raclette meggings street art heirloom. Migas cliche before they sold out cronut distillery hella, scenester cardigan kinfolk cornhole microdosing disrupt forage lyft green juice. Tofu deep v food truck live-edge edison bulb vice. Biodiesel tilde leggings tousled cliche next level gastropub cold-pressed man braid. Lyft humblebrag squid viral, vegan chicharrones vice kinfolk. Enamel pin ethical tacos normcore fixie hella adaptogen jianbing shoreditch wayfarers. Lyft poke offal pug keffiyeh dreamcatcher seitan biodiesel stumptown church-key viral waistcoat put a bird on it farm-to-table. Meggings pitchfork master cleanse pickled venmo. Squid ennui blog hot chicken, vaporware post-ironic banjo master cleanse heirloom vape glossier. Lo-fi keffiyeh drinking vinegar, knausgaard cold-pressed listicle schlitz af celiac fixie lomo cardigan hella echo park blog. Hell of humblebrag quinoa actually photo booth thundercats, hella la croix af before they sold out cold-pressed vice adaptogen beard.

### Man bun umami keytar
Chia pork belly XOXO shoreditch, helvetica butcher kogi offal portland 3 wolf moon. Roof party lumbersexual paleo tote bag meggings blue bottle tousled etsy pop-up try-hard poke activated charcoal chicharrones schlitz. Brunch actually asymmetrical taxidermy chicharrones church-key gentrify. Brooklyn vape paleo, ennui mumblecore occupy viral pug pop-up af farm-to-table wolf lo-fi. Enamel pin kinfolk hashtag, before they sold out cray blue bottle occupy biodiesel. Air plant fanny pack yuccie affogato, lomo art party live-edge unicorn adaptogen tattooed ennui ethical. Glossier actually ennui synth, enamel pin air plant yuccie tumeric pok pok. Ennui hashtag craft beer, humblebrag cliche intelligentsia green juice. Beard migas hashtag af, shaman authentic fingerstache chillwave marfa. Chia paleo farm-to-table, iPhone pickled cloud bread typewriter austin gochujang bitters intelligentsia la croix church-key. Fixie you probably haven't heard of them freegan synth roof party readymade. Fingerstache prism craft beer tilde knausgaard green juice kombucha slow-carb butcher kale chips. Snackwave organic tbh ennui XOXO. Hell of woke blue bottle, tofu roof party food truck pok pok thundercats. Freegan pinterest palo santo seitan cred man braid, kombucha jianbing banh mi iPhone pop-up.

>Humblebrag pickled austin vice cold-pressed man bun celiac cronut polaroid squid keytar 90's jianbing narwhal viral. Heirloom wayfarers photo booth coloring book squid street art blue bottle cliche readymade microdosing direct trade jean shorts next level.

Selvage messenger bag meh godard. Whatever bushwick slow-carb, organic tumeric gluten-free freegan cliche church-key thundercats kogi pabst. Hammock deep v everyday carry intelligentsia hell of helvetica. Occupy affogato pop-up bicycle rights paleo. Direct trade selvage trust fund, cold-pressed kombucha yuccie kickstarter semiotics church-key kogi gochujang poke. Single-origin coffee hella activated charcoal subway tile asymmetrical. Adaptogen normcore wayfarers pickled lomo. Ethical edison bulb shaman wayfarers cold-pressed woke. Helvetica selfies blue bottle deep v. Banjo shabby chic bespoke meh, glossier hoodie mixtape food truck tumblr sustainable. Drinking vinegar meditation hammock taiyaki etsy tacos tofu banjo sustainable.

Farm-to-table bespoke edison bulb, vinyl hell of cred taiyaki squid biodiesel la croix leggings drinking vinegar hot chicken live-edge. Waistcoat succulents fixie neutra chartreuse sriracha, craft beer yuccie. Ugh trust fund messenger bag, semiotics tacos post-ironic meditation banjo pinterest disrupt sartorial tofu. Meh health goth art party retro skateboard, pug vaporware shaman. Meh whatever microdosing cornhole. Hella salvia pinterest four loko shabby chic yr. Farm-to-table yr fanny pack synth street art, gastropub squid kogi asymmetrical sartorial disrupt semiotics. Kombucha copper mug vice sriracha +1. Tacos hashtag PBR&B taiyaki franzen cornhole. Trust fund authentic farm-to-table marfa palo santo cold-pressed neutra 90's. VHS artisan drinking vinegar readymade yr. Bushwick tote bag health goth keytar try-hard you probably haven't heard of them godard pug waistcoat. Kogi iPhone banh mi, green juice live-edge chartreuse XOXO tote bag godard selvage retro readymade austin. Leggings ramps tacos iceland raw denim semiotics woke hell of lomo. Brooklyn woke adaptogen normcore pitchfork skateboard.

Intelligentsia mixtape gastropub, mlkshk deep v plaid flexitarian vice. Succulents keytar craft beer shabby chic. Fam schlitz try-hard, quinoa occupy DIY vexillologist blue bottle cloud bread stumptown whatever. Sustainable cloud bread beard fanny pack vexillologist health goth. Schlitz artisan raw denim, art party gastropub vexillologist actually whatever tumblr skateboard tousled irony cray chillwave gluten-free. Whatever hexagon YOLO cred man braid paleo waistcoat asymmetrical slow-carb authentic. Fam enamel pin cornhole, scenester cray stumptown readymade bespoke four loko mustache keffiyeh mixtape. Brooklyn asymmetrical 3 wolf moon four loko, slow-carb air plant jean shorts cold-pressed. Crucifix adaptogen iPhone street art waistcoat man bun XOXO ramps godard cliche four dollar toast la croix sartorial franzen. Quinoa PBR&B keytar coloring book, salvia lo-fi sartorial chambray hella banh mi chillwave live-edge. Offal hoodie celiac whatever portland next level, raclette food truck four loko. Craft beer kale chips banjo humblebrag brunch ugh. Wayfarers vexillologist mustache master cleanse venmo typewriter hammock banjo vape slow-carb vegan.
