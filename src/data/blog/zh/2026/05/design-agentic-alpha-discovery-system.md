---
author: Yuhao Chen
pubDatetime: 2026-05-27T00:47:21.200Z
title: "Design agentic alpha discovery system"
slug: design-agentic-alpha-discovery-system
featured: false
draft: false
tags:
  - Technology
  - LLM
  - Agentic
description: "Design agentic alpha discovery system"
---

在准备写这篇文章的时候，我的心有点略微茫然：已经很久没有在独立自主，不依靠ai写过一整篇文章了。对我来说，这一次写作似乎变成了一种挑战。但是，我还是想记录一下关于这个项目的心路历程。

我一直期待能用非leetcode手段展示我自己的能力，这次的岗位要求正好非常具体以至于我能做出一个demo。当然，在我开始做这个project之前，我需要定下来一个scope。一方面，我需要展示自己对alpha research的了解。虽然以前有积累，本科的时候我曾经做过factor analysis，以及研究生就是对口的专业，但是毕竟很多年过去了记忆也有些模糊。感谢deepresearch和一系列ai研究工具，使得我能很快的从新的视角开始了解整个industry，以及具体公司是怎么做alpha research的。另一方面，这个project只是短期用来展示的mvp，那么我就需要决定我应该关注哪些要素，同时舍弃哪些要素。在这一点，我觉得我应该在传统的alpha discovery流程上强调agentic的部分。更具体一点，我的计划应该是分三个部分：首先是是把重心放在应该用llm来自动化的部分，其次解释不用llm而是deterministic的部分，最后省略的部分。这样子，我可以在一定的时间下完成demo的展示。

从理论上来讲，一个完整的alpha discovery应该包含以下的要素：产生想法（从一大堆数据中产生，或者是先有了想法再去找数据），然后根据想法构建可以测试的指标，之后做一系列包括回测在内的测试，最后在做eval。在这里面，我觉得最需要agentic能力的是前两部，并且对第一步自动化能带来最大的效益。原因是我觉得工业界对回测以及后续的步骤应该早就自动化了，现在更需要是自动产生结构化的假设和feature。data ingestion我觉得目前来看在不了解公司strategy的情况下也展示不了ai engineer的能力。

那么进一步思考，在这些步骤里，什么是应该让llm来输出的呢？什么是需要deterministic 或者 hard coded呢？众所周知，llm内生的幻觉问题，来自于本身预训练和后训练的局限性。那么我个人是不太敢用llm来review自己，或者让llm直接来计算结果的。所以，我通常是让llm plan并调用tools，最终产生structure output。然后一些python script直接读取json，来执行idea -raw data - feature的产生。这么做优点是可控性高，在这个demo里我事先定义好了有限的data operation，以及检索/profiling的 tool。这样agent可以在获取确定信息的情况下，自由的思考和propose，然后又输出可控的结果。在这个框架下，我让claude code提出了implementation plan，并继续修改，最后用codex来执行这个计划。这么做是因为cc太贵了，codex相对比较便宜，同时api我也用的openrouter免费的nvdia nemotron。

到了这里，基本上这个demo就到尾声了，在结束这篇文章之前我还想提出如果要把这个系统放在production还需要注意什么。scalability应该是我最先考虑的，这个系统存在意义应该是如何让一堆agent同时产生idea并且同时测试，那么现在这种简单的设计肯定是不行的，需要加入并行处理，以及一些容错的机制。同时，我也需要考虑researcher是怎么和这个系统结合的，是通过修改prompt，还是跟我们ai engineer合作并提供需求。目前单纯让llm从数据中产生hypothesis还是太naive了。最后，这样的agentic system貌似需要memory和自我提升，不过目前来讲这些还是属于prompt engineer/context engineer/harness engineer的内容。

作为结尾，感谢大家能阅读玩这篇纯粹手写的文章。我自己读完其实都有奇怪的感觉，一方面是对自己能写出和现在ai批量生产的文章不同的欣慰，但是一方面我觉得我仍旧怀疑人类写手产出的文章质量能比得上ai吗。不过不管怎么样，以后我会把写作当作一种持续的练习。


