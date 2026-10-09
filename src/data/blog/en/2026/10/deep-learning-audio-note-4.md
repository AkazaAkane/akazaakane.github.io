---
author: Yuhao Chen
pubDatetime: 2026-10-09T19:40:00.000Z
title: "Deep Learning Audio Note 4: Language, Knowledge, and Deployment in ASR"
featured: false
draft: false
tags:
  - Technology
  - Audio
  - Deep Learning
  - ASR
description: "Connect the five dimensions of ASR through language model fusion, knowledge sources, and the constraints of offline, streaming, and on-device deployment."
---

This is the last post in this round of ASR notes. After several consecutive days of writing, I was getting a little tired, so this draft started with AI-generated text that I then rewrote and organized.

The [previous post](/en/posts/deep-learning-audio-note-3/) explored Alignment: how continuous speech maps to discrete text. This post covers Language, Knowledge, and Deployment, connecting all five dimensions:

**ASR = Representation + Alignment + Language + Knowledge + Deployment**

This is an analytical framework for understanding systems, rather than five mandatory modules executed in sequence.

[阅读中文版](/zh/posts/deep-learning-audio-note-4/)。

## Language: when several words sound alike, which makes sense?

Alignment connects positions in speech to output tokens. Even with a valid alignment, acoustic evidence alone may not determine the transcript.

Consider `two / to / too` in:

```text
I want ___ go home
```

Their pronunciations can be identical or very similar, but language regularities favor `to`. A text-only language model estimates:

```text
P(yᵤ | y<ᵤ)
```

This is the probability of the next token given preceding text. ASR must also use audio X. Language priors cannot replace acoustic evidence: otherwise, a recognizer may rewrite a real but uncommon expression into a familiar sentence.

Language modeling has several routes. A hand-written grammar can specify `COMMAND → ACTION OBJECT`. An n-gram LM estimates probabilities from a limited history, such as `P(wₜ | wₜ₋₂, wₜ₋₁)`. Neural LMs, including LSTMs and Transformers, can use longer histories. More regularities have become learned from data, while explicit grammars and finite-state constraints remain useful for particular tasks.

### How does language enter ASR?

ASR architectures use text history in different ways.

CTC factorizes path probabilities across time given the audio, without an explicit autoregressive text state. Its encoder can still learn language-related regularities from paired training data. An external LM can score candidates during beam search or rerank them afterward:

```text
CTC frame scores + optional LM scores
                 ↓
          Beam search → hypotheses
                 ↓
          Optional LM rescoring
```

LAS / AED directly conditions its decoder on previous text and acoustic context:

```text
P(Y | X) = ∏ᵤ P(yᵤ | y<ᵤ, X)
```

[RNN-T](https://arxiv.org/abs/1211.3711) organizes audio and text history into two information streams:

```text
Audio → Encoder ──────────┐
                          ├→ Joiner → token / blank
Text history → Predictor ─┘
```

The predictor represents previously emitted non-blank tokens, and the joiner combines this with encoder output. RNN-T explicitly handles acoustic information, monotonic alignment, and text history. Its predictor has an LM-like role, but is not equivalent to an independently trained LM; some variants also restrict its history. An external LM can still be added.

### Rescoring and fusion

One direct use of an external LM is to generate several hypotheses, then rerank them. A representative score is:

```text
S(Y) = log P_ASR(Y | X) + λ log P_LM(Y) + β len(Y)
```

λ controls the LM weight; β is an optional length or insertion reward whose definition depends on the decoder. Rescoring cannot recover a correct transcript that is absent from the candidate set.

Weighted scores can also guide search itself. The integration point matters:

| Method                                          | How the LM participates                                                                                     |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Rescoring                                       | Scores N-best candidates or lattice paths after first-pass recognition                                      |
| Shallow fusion                                  | Combines ASR and LM scores during decoding, influencing candidate expansion                                 |
| [Deep fusion](https://arxiv.org/abs/1503.03535) | Combines hidden representations from separately trained decoders and LMs, then trains the fusion components |
| [Cold fusion](https://arxiv.org/abs/1708.06426) | Introduces a pretrained, typically frozen LM during Seq2Seq training, with gates that learn how to use it   |

These names come from sequence generation research; ASR implementations can vary. The distinctions include when the LM joins the system, whether scores or representations are combined, and which parameters are trained.

More language capability inside ASR does not eliminate external LMs. Additional text, domain vocabulary, and decoding constraints can still help. Their value should be measured through recognition errors and unwanted corrections in the target setting.

## Knowledge: where do these capabilities come from?

Here, Knowledge means **sources of knowledge and how it is acquired**, rather than factual question-answering ability. It is an analytical dimension in this article, not a standard ASR component strictly separate from Language.

Language asks what linguistic regularities a model knows. Knowledge asks how it learned them.

Preferring `I want to go home` over `I want two go home` reflects a language regularity. That preference might come from a grammar, labeled transcripts, or large text corpora. Pronunciation lexicons provide word-to-pronunciation mappings; unlabeled speech, self-supervised objectives, weakly supervised pairs, and pseudo-labels provide different learning signals.

Knowledge therefore affects several dimensions:

```text
                Knowledge
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
Representation  Alignment   Language
```

### From expert structure to learning from data

Traditional systems contain substantial human structure: phoneme inventories, lexicons, HMM topologies, and task grammars. In GMM-HMM systems, people define structures while data supplies acoustic distributions, transition probabilities, and statistics such as n-gram probabilities.

DNN-HMM systems use neural networks to provide acoustic state posteriors, converted into decoding scores through operations such as prior correction. Lexicons, state structures, and external LMs remain. Deep learning did not suddenly remove human-designed structure.

End-to-end methods such as CTC, RNN-T, and LAS let neural networks learn mappings from audio to output tokens, reducing dependence on explicit phoneme states and pronunciation lexicons:

```text
Traditional system:
features + acoustic scores + HMM + lexicon + LM → text

End-to-end system:
audio → neural sequence model → text tokens
```

This is a structural simplification. Traditional components often participate in a joint search rather than a mechanical pipeline. End-to-end systems still require choices about tokenization, networks, alignment constraints, and objectives. CTC, RNN-T, and AED also model output dependencies differently; they do not all contain the same kind of internal LM.

### Self-supervision changes knowledge acquisition

[wav2vec 2.0](https://arxiv.org/abs/2006.11477) learns representations from unlabeled speech before ASR fine-tuning with transcribed audio. Its key contribution concerns knowledge acquisition, rather than a new alignment mechanism replacing CTC.

These concepts describe different dimensions:

| Concept                        | What it primarily describes                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| Transformer / Conformer        | Network architecture                                                                              |
| CTC                            | A training objective with a blank-collapse alignment rule                                         |
| RNN-T                          | A sequence model and objective with a predictor, joiner, and marginalization over monotonic paths |
| Self-supervised learning (SSL) | A strategy for constructing learning signals from data and learning representations               |

[Whisper](https://arxiv.org/abs/2212.04356) uses a Transformer encoder-decoder trained on roughly 680,000 hours of multilingual, multitask weakly supervised audio. It demonstrates the value of scale and data diversity for generalization across datasets, rather than inventing a new basic architecture. Generalization does not establish reliability for every accent, noise condition, or specialist domain.

Foundation model training signals can come from paired audio and text, unlabeled speech, web text, weak labels, pseudo-labels, synthetic data, and multilingual or multimodal data. Architecture alone cannot explain capability differences: data quality, coverage, objectives, and training scale also matter.

## Deployment: under what constraints must the model run?

Representation, Alignment, and Language explain how speech becomes text. Deployment asks under what practical conditions this must happen, shaping model design and training.

### Offline ASR

Offline recognition can process the complete input `X = x₁:T`, allowing bidirectional context, global attention, larger searches, and second-pass rescoring.

It can spend more computation on accuracy, but still faces throughput, memory, cost, and completion-time constraints. Long recordings may require segmentation because of memory limits.

### Streaming ASR

Streaming recognition produces results as audio arrives. A causal model uses available audio; a model with limited right context can be written as:

```text
hₜ = f(x₁:ₜ₊ᵣ)
```

Computing hₜ requires waiting for the necessary r future frames. More context may improve recognition, at the cost of waiting. Quality gains are not necessarily monotonic and depend on training and data.

RNN-T's monotonic paths support incremental decoding, but **streaming also requires a causal encoder or bounded future context**. The original RNN-T paper discusses bidirectional networks too. CTC and AED designed for streaming can also support online recognition.

Deployment shapes the encoder. Standard global self-attention has quadratic attention computation in sequence length T. Common optimizations include chunked or local attention, bounded right context, causal convolutions, and stronger subsampling.

[FastConformer](https://arxiv.org/abs/2305.05084) improves efficiency through changes including subsampling, and studies limited-context attention for long recordings. **Support for long recordings does not establish low-latency streaming**: right context, convolutional access to future frames, caching, and chunking still need inspection.

### Streaming latency is more than compute time

For a particular result being committed, a rough diagnostic model is:

```text
L ≈ L_buffer/context + L_compute + L_queue + L_commit/endpoint
```

- **Buffer / context**: waiting for chunks or required future context.
- **Compute**: encoder, decoder, and related computation.
- **Queue**: waiting for batching or scheduling under concurrency.
- **Commit / endpoint**: waiting for stable output, commitment, or detection that speech has ended.

These components can overlap. End-to-end experience may also include capture, transport, and display time. An evaluation should specify whether it measures the first partial result, stable text, or final output after speech ends.

Real-Time Factor (RTF) generally divides processing time by audio duration. Low RTF indicates fast processing under the measured conditions, but does not guarantee low first-token latency, commitment delay, or tail latency under load.

### Edge ASR: balancing several objectives

Phones, cars, and earbuds add memory, compute, power, battery, thermal, and privacy constraints. The goal becomes minimizing WER within resource and latency budgets.

CTC greedy decoding is simple:

```text
Encoder → vocabulary projection → argmax → collapse
```

Collapse first merges consecutive repeated symbols, then removes blanks. A simple decoder can reduce overhead, but the encoder may dominate total cost. The CTC label alone does not establish suitability for a device.

RNN-T adds text-history modeling and sequential decoding computation, providing a way to combine history dependencies with monotonic incremental output. Its suitability needs comparison on the same data, hardware, and latency budget. Quantization, model size, caching, search width, and runtime implementation also affect the outcome.

## Connecting all five dimensions

The framework can now be written as:

```text
Speech → Representation H → Alignment + Language → Text Y
```

- **Representation**: how should sound be represented?
- **Alignment**: how does continuous speech map to discrete text?
- **Language**: when candidates sound plausible, which text sequence makes sense?
- **Knowledge**: which data, human structures, and learning signals supply these capabilities?
- **Deployment**: which latency, throughput, and resource constraints must the system satisfy?

The last two dimensions influence the whole system; Alignment and Language act together in many models:

```text
                 Knowledge sources / training signals
                                  ↓
Speech → Representation → [Alignment + Language] → Text
                                  ↑
                       Deployment constraints
```

Long-term trends include learning more from data, training components jointly, and considering data scale alongside deployment design. These approaches coexist with explicit constraints, modular systems, and external LMs.

From GMM-HMM and DNN-HMM to CTC, RNN-T, LAS, Whisper, and speech foundation models, the change is more than increasing model size. Representations, alignment, output dependencies, knowledge sources, and practical operating constraints have all evolved together.
