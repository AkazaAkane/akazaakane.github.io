---
author: Yuhao Chen
pubDatetime: 2025-06-04T05:00:00.000Z
modDatetime: 2025-06-04T21:57:12.990Z
title: "Minimax Speech 2.0"
slug: minimax-speech-2-0
featured: false
draft: false
tags:
  - 技术
  - 论文
  - Minimax
description: "Minimax Speech 2.0 论文阅读笔记：语音合成模型的技术细节"
---

原文链接 [https://arxiv.org/pdf/2505.07916](https://arxiv.org/pdf/2505.07916)

## 导言

我个人对Minimax这个公司还是比较友好感的：之前听过几次他们ceo和cto的podcast，能感受到他们不仅有商业上的布局，在技术上也有坚定的追求（linear attention）。所以，我对他们的新模型还是蛮期待的。事实证明，这一次新的tts模型用起来确实很优秀，尤其是在中文语音中。不过这篇技术报告内容一般，一半以上的篇幅都在讲自己效果怎么怎么好，感觉目的可能是秀肌肉居多而不是分享，猜测可能公司有融资方面的压力。

Minimax Speech 2.0是一个自回归的transformer架构tts（Text to Speech）模型，并且达到了sota的结果。这个模型的创新点在于，运用了一个科学系的speaker encoder使得0-shot learning成为可能，并且有也支持one shot。不仅如此，模型还运用了flow matching和flow vae decoder使得生成的效果更好。

## 数据

很可惜，这篇技术报告并没有提到训练数据的细节，只是含糊的讲了大家都知道的一些数据组成和预处理的方法：训练用了32种语言的数据；采用了两个独立地向ASR(Auto Speech Recognition)模型进行音频转录，加入结果接近可以认为是准确的，否则进一步将处理；用VAD（Voice Activity Detection）配合asr输出时间戳以及标点符号；保留录音中的背景稳态噪声，提高模型在真实环境下的鲁棒性；用SVR（Speaker Verification Model。目前我印象里比较全的TTS模型数据的描述还是来自几年前的open-ai whisper，感觉国内厂商在这一方面还是比较保守。

## 模型结构

模型的架构是经典的多模态架构：分别将不同模态压缩到一个unified space，然后decode出output。具体来讲，文字是用了经典的bpe作为encoder，语音则是用了Speaker Encoder + Audio Tokenizer，一个用来提取声音特征一个用来提取内容。与其他tts模型不同的是，minimax没有用一个pre-trained的audio encoder，而是把这个encoder和ar transformer用来一起训练。这么做的优点在于，pre-trained encoder的语料数据不够丰富，个人猜测可能对于中文的效果不好，minimax这一次应该是在数据方面加强了中文语料。

架构上的创新使得minimax可以实现高质量的0-shot learning，也就是用户只需要上传一段reference的语音就可以直接通过文字输出想要的声音克隆片段。相比之下，传统的语音模型需要 语音-文本 对进行 1-shot或者fine-tuning 才能达到不错的效果。

## flow matching

Flow Matching模型是一种生成模型，本质是学习一种连续变换将简单的分布变成复杂的连续分布，tts模型一般会把ar transformer生成的离散token转换成连续的分布。

---

### **1\. 自回归 Transformer：生成离散音频 tokens**

-   **输入条件**：
    
    -   文本编码后的 tokens（记为 ）。
    -   说话人编码器输出的条件向量（记为 ），用于指定目标说话人的音色和风格。
-   **处理逻辑**：  
    自回归 Transformer 以文本 tokens 为输入，结合说话人条件向量 ，通过注意力机制逐步生成离散音频 tokens（记为 ）。这一过程模仿人类语音生成的时序特性，擅长捕捉韵律和语调的自然变化。
    
-   **优势**：  
    相比非自回归模型，自回归架构无需显式建模音素持续时间对齐，通过隐式学习生成更自然的语音节奏。
    

---

### **2\. Latent Flow Matching 模块：从离散 tokens 到连续语音特征**

自回归 Transformer 输出的离散音频 tokens 随后进入 Latent Flow Matching 模块，该模块包含两个关键组件：

#### **(1) Flow-VAE：优化潜在特征表示**

-   **结构与功能**：
    
    -   **Encoder**：将离散音频 tokens 转换为连续语音特征（潜在变量 ），捕捉音频的声学细节（如音高、音色）。
    -   **Flow Model**：对潜在变量 的分布进行可逆变换，将其映射到标准正态分布，以增强特征的表达能力和分布拟合能力。
    -   **Decoder（神经声码器）**：将潜在变量 还原为音频波形 ，通过 KL 散度约束确保重建精度。
-   **创新点**：  
    传统 VAE 假设潜在空间为标准正态分布，而 Flow-VAE 通过流模型的可逆变换（如仿射变换、置换），学习更复杂的后验分布，从而更准确地捕捉语音数据的多模态特征。  
    实验表明，Flow-VAE 的波形重建误差低于传统 VAE，且生成的语音特征更紧凑、信息更丰富。
    

---

#### **(2) 流匹配模型（Flow Matching Model）**

-   **输入条件**：
    
    -   自回归 Transformer 生成的离散音频 tokens （经 Flow-VAE 编码为潜在变量 ）。
    -   说话人条件向量 和文本编码后的上下文信息 （用于引导合成语音的风格和内容对齐）。
-   **处理逻辑**：  
    流匹配模型基于 Transformer 架构，对潜在变量 的分布进行建模，通过匹配数据分布与先验分布（如标准正态分布），生成高质量的连续语音特征。该过程无需显式建模时长，而是通过隐式学习捕捉语音的时序依赖。
    
-   **优势**：  
    相比直接预测下一个 token（Next Token Prediction），流匹配模型通过连续潜在空间的分布建模，避免了离散空间的量化误差，且能更灵活地处理语音的动态范围和细节变化。