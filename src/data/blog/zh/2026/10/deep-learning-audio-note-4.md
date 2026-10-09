---
author: Yuhao Chen
pubDatetime: 2026-10-09T19:40:00.000Z
title: "深度学习音频笔记 4：ASR 的语言、知识与部署"
featured: false
draft: false
tags:
  - Technology
  - Audio
  - Deep Learning
  - ASR
description: "从语言模型融合、知识来源到离线、流式与端侧约束，把 Representation、Alignment、Language、Knowledge 和 Deployment 五个维度连起来。"
---

这是这一轮 ASR 笔记的最后一篇。连续写了几天，有些累了，这次的初稿由 AI 生成，再由我改写整理。

[上一篇](/zh/posts/deep-learning-audio-note-3/)讨论了 Alignment：连续声音怎样对应离散文字？这一篇继续讨论 Language、Knowledge 和 Deployment，把前面提出的五个维度连起来：

**ASR = Representation + Alignment + Language + Knowledge + Deployment**

这是一个帮助理解系统的分析框架，并不是五个必须独立实现、依次执行的模块。

[Read this article in English](/en/posts/deep-learning-audio-note-4/)。

## Language：听起来像多个词时，哪个更合理？

Alignment 解决的是：声音中的哪些位置，对应输出的哪些 token？但即使对齐没有问题，仅靠声学证据也不一定能确定转写。

最典型的例子是 `two / to / too`。在下面的上下文中：

```text
I want ___ go home
```

这些词的发音可能相同或非常接近，但语言规律会让 `to` 成为更合理的选择。纯语言模型关心的是：

```text
P(yᵤ | y<ᵤ)
```

也就是给定此前文字，下一个 token 有多大概率出现。ASR 则还要结合音频 X；语言先验不能替代声学证据，否则模型也可能把真实但少见的表达改成常见句子。

语言建模有多条路线：人工 grammar 可以规定 `COMMAND → ACTION OBJECT`，n-gram LM 从语料估计有限历史下的词概率，例如 `P(wₜ | wₜ₋₂, wₜ₋₁)`；LSTM、Transformer 等神经语言模型则可以利用更长的历史。这里的趋势是把更多语言规律交给数据学习，人工语法和有限状态约束仍然可以用于特定任务。

### Language 如何进入 ASR？

不同 ASR 架构对文字历史的使用方式不同。

CTC 在给定音频的条件下，将路径概率分解为各时间步概率的乘积，没有显式的自回归文字状态。它的 encoder 仍可能从配对语料学到语言相关规律；“没有显式文字历史”并不意味着完全没有语言信息。外部 LM 可以在 beam search 中参与打分，也可以重排候选：

```text
CTC frame scores + optional LM scores
                 ↓
          Beam search → hypotheses
                 ↓
          Optional LM rescoring
```

LAS / AED 则在 decoder 中直接使用此前文字和声学上下文：

```text
P(Y | X) = ∏ᵤ P(yᵤ | y<ᵤ, X)
```

[RNN-T](https://arxiv.org/abs/1211.3711)把声学输入与文字历史组织成两条信息流：

```text
Audio → Encoder ──────────┐
                          ├→ Joiner → token / blank
Text history → Predictor ─┘
```

Predictor 根据此前输出的非 blank token 生成历史表示，再与 encoder 输出结合。因此 RNN-T 同时显式处理声学信息、单调对齐和文字历史。不过 Predictor 只是具有类似语言模型的作用，并不等价于独立训练的 LM；一些变体也使用受限历史。RNN-T 仍然可以融合外部 LM。

### Rescoring 与 fusion

一种直接的外部 LM 用法，是先生成多个候选，再重新排序。常见的示意评分为：

```text
S(Y) = log P_ASR(Y | X) + λ log P_LM(Y) + β len(Y)
```

λ 控制 LM 权重，β 是可选的长度或插入奖励；具体定义取决于解码器。Rescoring 只能重排已有候选，如果正确转写没有进入候选集，就不能靠这个步骤恢复。

同样的加权思想也可以用于搜索过程，但接入位置不同：

| 方法                                            | LM 如何参与                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| Rescoring                                       | 在第一遍识别之后，对 N-best 候选或 lattice 中的路径重新打分        |
| Shallow fusion                                  | 在解码搜索时组合 ASR 与 LM 的分数，影响候选扩展                    |
| [Deep fusion](https://arxiv.org/abs/1503.03535) | 组合单独训练的 decoder 与 LM 的隐藏表示，再训练融合部分            |
| [Cold fusion](https://arxiv.org/abs/1708.06426) | 从 Seq2Seq 训练阶段就引入预训练、通常冻结的 LM，通过门控学习使用它 |

这些 fusion 名称源于序列生成研究，具体 ASR 实现可能有所不同。它们的区别不只是“有没有外部 LM”，还包括何时接入、组合分数还是表示、哪些参数参与训练。

更多语言能力进入 ASR 内部，并不意味着外部 LM 消失。额外文本、领域词汇和解码约束仍然可能有价值，收益需要用目标场景中的错误率和误纠正情况验证。

## Knowledge：模型的这些能力从哪里来？

这里的 Knowledge 指**知识来源与获取方式**，不特指事实问答知识。它不是一个与 Language 严格分离的标准 ASR 模块，而是本文的分析维度。

Language 问的是：模型掌握什么语言规律？Knowledge 问的是：这些能力怎么获得？

例如，模型认为 `I want to go home` 比 `I want two go home` 更合理，这是语言规律。而这些规律可能来自人工 grammar、标注转写或大规模文本。发音词典提供词与发音之间的映射；未标注语音、自监督目标、弱监督配对和伪标签，则提供不同的学习信号。

因此 Knowledge 同时影响多个维度：

```text
                Knowledge
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
Representation  Alignment   Language
```

### 从专家知识到数据学习

传统系统包含大量人工结构，例如音素集合、发音词典、HMM 拓扑和任务语法。在 GMM-HMM 系统中，人定义结构，数据估计声学分布、状态转移以及 n-gram 等统计量。

DNN-HMM 用神经网络提供声学状态后验，并通过先验校正等方式供 HMM 解码使用。它保留了词典、状态结构和外部 LM 等组成部分。深度学习并没有突然消灭人工结构。

CTC、RNN-T 和 LAS 等端到端方法，则让神经网络直接学习音频到输出 token 的映射，减少对显式音素状态和发音词典的依赖：

```text
Traditional system:
features + acoustic scores + HMM + lexicon + LM → text

End-to-end system:
audio → neural sequence model → text tokens
```

这只是结构上的简化。传统解码的组件往往联合搜索，并非机械串联；端到端系统也仍然包含 tokenizer、网络结构、对齐约束和训练目标等设计选择。不同方法对语言依赖的建模也不同，不能把 CTC、RNN-T 和 AED 都理解为带有相同的内部 LM。

### 自监督学习改变知识获取方式

[wav2vec 2.0](https://arxiv.org/abs/2006.11477)先从未标注语音学习表示，再用带转写的语音进行 ASR 微调。它的重要贡献是知识获取方式，而不是提出一种替代 CTC 的新对齐机制。

几个概念应放在不同维度理解：

| 概念                            | 主要描述什么                                                |
| ------------------------------- | ----------------------------------------------------------- |
| Transformer / Conformer         | 网络架构                                                    |
| CTC                             | 带有 blank-collapse 对齐规则的训练目标                      |
| RNN-T                           | 包含 predictor、joiner 和单调路径边缘化的序列模型及训练目标 |
| Self-supervised learning（SSL） | 从数据构造训练信号、学习表示的策略                          |

[Whisper](https://arxiv.org/abs/2212.04356)采用 Transformer encoder-decoder，使用约 68 万小时多语言、多任务的弱监督音频数据训练。它展示了训练规模与数据多样性对跨数据集泛化的价值，而不是靠发明一个全新的基础架构。泛化能力也不等于在每个口音、噪声环境或专业领域都可靠。

基础模型的训练信号可以来自配对音频与文本、未标注语音、网页文本、弱标签、伪标签、合成数据，以及多语言、多模态数据。仅比较架构，已经不足以解释两个模型的能力差异；数据质量、覆盖范围、目标设计和训练规模都要一起看。

## Deployment：模型最终要在什么约束下运行？

Representation、Alignment 和 Language 讨论怎样把语音变成文字；Deployment 则问：在什么现实条件下完成这件事？它会反过来限制模型设计和训练方式。

### Offline ASR

离线识别可以在完整音频到达后处理 `X = x₁:T`，因此有条件使用双向上下文、全局 attention、更大的搜索范围和第二遍重评分。

它可以为准确率投入更多计算，但离线并不等于没有性能约束。长音频转写仍然要考虑吞吐、显存、成本和任务完成时间；内存限制也可能要求分段处理。

### Streaming ASR

流式识别在音频到达的同时产生结果。纯因果模型只能使用已到达的音频；如果模型需要有限的右侧上下文，可以写成：

```text
hₜ = f(x₁:ₜ₊ᵣ)
```

要计算 hₜ，就必须等到所需的 r 帧未来上下文到达。更多上下文可能提高识别质量，但也会增加等待时间；收益并不保证单调增长，还取决于训练和数据。

RNN-T 的单调路径适合增量解码，但**流式能力还取决于 encoder 是否因果、是否限制未来上下文**。原始 RNN-T 论文也讨论了双向网络，不能仅凭名称就称任何 RNN-T 为天然流式模型。CTC 和经过流式设计的 AED 同样可以用于在线识别。

Encoder 的计算方式也受到部署影响。对长度为 T 的序列，标准全局 self-attention 的注意力计算随 T 呈二次增长。常见优化包括 chunked / local attention、有限右侧上下文、因果卷积，以及更强的下采样。

[FastConformer](https://arxiv.org/abs/2305.05084)通过重新设计下采样等部分提高效率，并研究有限上下文 attention 处理长音频。**长音频可运行不等于低延迟流式可运行**：还要检查右侧上下文、卷积是否使用未来帧、缓存和分块策略。

### Streaming latency 不只是模型算得快不快

对某个待提交的识别结果，可以用下面的式子粗略梳理延迟：

```text
L ≈ L_buffer/context + L_compute + L_queue + L_commit/endpoint
```

- **Buffer / context**：等待音频分块或所需未来上下文。
- **Compute**：encoder、decoder 等计算耗时。
- **Queue**：并发时等待 batching 或 scheduler 的时间。
- **Commit / endpoint**：等待结果稳定、确认提交或判断说话结束。

这是诊断框架；各部分可能重叠，端到端体验还可能包括采集、传输和显示时间。评估时应说明测的是首个部分结果、稳定文字，还是说话结束后的最终结果。

RTF（Real-Time Factor）通常是处理耗时与音频时长之比。RTF 很低，说明在相应测试条件下处理得快，却不能保证首字延迟、提交延迟或高并发下的尾延迟也低。

### Edge ASR：多目标权衡

手机、汽车、耳机等设备还要考虑内存、算力、功耗、电池、散热和隐私。目标因此是：在资源和延迟预算内，尽量降低 WER，而不是孤立地追求最低错误率。

CTC greedy decoding 的路径很简单：

```text
Encoder → vocabulary projection → argmax → collapse
```

其中 collapse 先合并连续重复符号，再移除 blank。解码器简单有利于减少开销，但系统总成本仍可能主要来自 encoder；CTC 标签本身也不能保证适合端侧。

RNN-T 加入文字历史和串行解码计算，换来一种结合历史依赖与单调增量输出的建模方式。它是否更适合某个设备，需要在同样的数据、硬件和延迟预算下比较。量化、模型大小、缓存、搜索宽度及运行时实现，也会改变结果。

## 把五个维度连起来

现在可以把整个框架写得更准确：

```text
Speech → Representation H → Alignment + Language → Text Y
```

- **Representation**：声音应该怎样表示？
- **Alignment**：连续声音怎样对应离散文字？
- **Language**：多个候选都听起来合理时，哪个文字序列更合理？
- **Knowledge**：这些能力从哪些数据、人工结构与训练信号中获得？
- **Deployment**：系统必须满足哪些延迟、吞吐和资源约束？

后两个维度贯穿整个系统，而 Alignment 和 Language 在许多模型中共同作用：

```text
                 Knowledge sources / training signals
                                  ↓
Speech → Representation → [Alignment + Language] → Text
                                  ↑
                       Deployment constraints
```

ASR 的长期趋势包括更多地从数据学习、让多个组件联合训练，以及同时考虑数据规模与部署设计。这些路线共存，并没有消除人工约束、模块化系统或外部 LM。

从 GMM-HMM、DNN-HMM，到 CTC、RNN-T、LAS，再到 Whisper 和语音基础模型，变化不只是模型越来越大。我们对声音表示、对齐、语言依赖、知识来源，以及实际运行约束的处理方式，都在同时演进。
