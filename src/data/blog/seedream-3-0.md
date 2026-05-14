---
author: Yuhao Chen
pubDatetime: 2025-04-21T05:00:00.000Z
modDatetime: 2025-04-22T02:17:54.000Z
title: "Seedream 3.0 Technical Report"
slug: seedream-3-0
featured: false
draft: false
tags:
  - 技术
  - 豆包
  - 论文
description: "Seedream 3.0 技术报告阅读笔记"
---

原文链接 [https://arxiv.org/pdf/2504.11346](https://arxiv.org/pdf/2504.11346)

## 导言

Seedream 2.0 虽然已经很好了，但是还有一些问题：模型在复杂prompt上的对齐有待提高，尤其是在数字精度和多物体空间关系的情况下；2.0对于图片内文字的生成能力有待提升；图片美学上的问题；以及生成图片的清晰度问题。对于以上问题，豆包做了一下提升：在数据层面，引入了双倍的高质量数据；增加了训练步骤和技巧，比如混合resolution training，多模态rope，新的representation alignment loss，以及resolution aware sampling。最后，也对后训练和生成加速做了提升。总而言之，3.0是一次对于2.0的incremental change，但是仅仅才过了一个月。从这能看出来字节豆包组的含金量，以及好的ai infra对于持续research和产品迭代的重要性。

## 数据

文章提到，2.0阶段运用了严格的数据筛选机制，所以这限制了训练数据的数量。在3.0中，豆包运用了新的筛选机制，把defect小于20%的数据保留下来，并且在训练的时候运用了spatial attention mask使得这些区域会被排除出训练，在保证模型稳定性的情况下成功扩展了差不多20%的训练数据。

## 预训练

模型结构沿用了2.0，只是增加了训练参数和以下的技巧：  
1.混合清晰度（resolution）训练。具体的讲，因为transformer天然支持不同长度的sequence，豆包组先用 256^2 的数据做pre-train，然后再更高清晰度（512^82 to 2048^2）的数据上做微调。并且，额外添加了size embedding作为额外的condition（应该是做了crocs attention？），使得模型能在没见过的清晰度情况下依然表现出色。  
2.Cross-modality Rope。在2.0中，运用的是scaling rope。在3.0中，对于这个技巧做了提升。以往我们会对text做1d rope，对图片做2d。但是在cm rope里，会把text也当作一维的2d，做2d rope并投射到2d空间和图片关联起来。  
3.运用了flow matching的损失函数，并且增加了alignment loss （用来对齐自己的mmdit和dinov2）可以让加速模型收敛。  
4.Resolution-aware Timestep Sampling是一项diffusion模型训练的技巧，原理是在不同的resolution下对于我们sample的distribution做改变：high resolution图片会让sampling dist更偏向于lower snrs/higher noise levels。在训练阶段是用数据集的平均的resolution，inference的时候用期望的resolution来决定shift factor。具体做法是先从log-normal sample，然后根据我们算出来的shift factor做shifting。

## 后训练

相比于2.0，3.0取消的refiner阶段因为模型本身已经能够生成不同resolution的图片。除此之外，还做了以下提升：为了ct和sft的阶段训练了更多的captioning model，能更好的让模型理解prompt中的美学，style和layout；平衡了数据里不同resolution数据的数量。

还有一点是用了vlm而不是clip作为奖励函数，具体做法如下：  
1.Instruction as Query: The model receives a prompt, such as “A cat sitting on a couch.”​  
2.Formulating the Question: This prompt is transformed into a question like, “Does this image depict a cat sitting on a couch? Please answer Yes or No.“  
3.Evaluating with VLM: The VLM processes the generated image and the question, outputting probabilities for “Yes” and “No.”​  
4.Deriving the Reward: The probability assigned to “Yes” is normalized and used as the reward signal. A higher probability indicates better alignment between the image and the prompt.

### 模型加速

seedream3.0的模型加速基于Hyper-SD和RayFlow，相比于传统的扩散模型在降噪过程中所有的样本都是通过一样的高斯分布路径，seedream对不同样本实现了个性化的单一通路，提升了模型的稳定性和生成的多元化。并且使用了一个预训练的模型来对噪声进行预估，这个方法使得模型在加噪和去噪的过程中能以最大可能性进行收敛，使得模型现在可以用较少的步数得到非常好的结果。在训练加速上，训练了一个结合Stochastic Stein Discrepancy (SSD)的neural net来预测哪个timestamp会产生最大的training loss，所以在训练采样的时候区别于传统的uniform sampling可以更高效的采样most important timesteamps。结合以上的工作，豆包的模型得以更高的效率达到普通扩散模型采样50步才能到达的效果。