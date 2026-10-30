---
author: Yuhao Chen
pubDatetime: 2026-10-30T04:21:26.000Z
timezone: America/New_York
title: "Deep Learning Audio Note 5: Keyword Spotting and Always-On Listening"
featured: false
draft: false
tags:
  - Technology
  - Audio
  - Deep Learning
  - KWS
description: "Understand lightweight keyword spotting models, Google and Apple cascade systems, streaming caches, robust training, and on-device optimization through the trade-off between false alarms and missed triggers."
---

This post covers KWS, following [Deep Learning for Audio — Lecture 4](https://github.com/severilov/DL-Audio-AIMasters-Course/blob/main/lectures/lecture04_DLAudio2026.pdf). The previous posts explored how sound becomes text. This time the question is: **how does a device find the phrase that should start an interaction in a continuous stream of sound?**

The discussion follows four threads: the task and its evaluation, model architectures, cascade systems, and deployment. DTW and HMMs provide background connecting this post to the earlier ASR notes. The subsequent models and systems follow the lecture; simplified formulas help explain their mechanisms.

[阅读中文版](/zh/posts/deep-learning-audio-note-5/)。

## What does KWS detect?

**Keyword spotting (KWS)** lets a device continuously listen for a phrase such as “Hey Siri,” accurately and with low power consumption. The main constraints are accuracy, latency, memory, and power. In the lecture's client-server design, the device detects a candidate wake word, then sends the trigger phrase and following query to a server for verification and ASR. The always-on detector must be lightweight and robust; later checks can spend more computation. A product can also keep more processing on the device.

| Application               | Input and output                             | Example                                     |
| ------------------------- | -------------------------------------------- | ------------------------------------------- |
| Device activation         | Continuous audio → presence of a wake phrase | Start an interaction after “Hey Siri”       |
| Recording search          | Long recording → keyword locations           | Find mentions of “refund” in a support call |
| Short-word classification | Short recording → word category              | Classify a one-second clip as “stop”        |

Wake word, hotword, and trigger word commonly refer to activation phrases. A typical assistant pipeline is on-device candidate detection → precise verification → ASR of the following request → execution. In “Hey Siri, what will the weather be tomorrow?”, KWS finds the activation phrase, while ASR recognizes the question.

## How should KWS be evaluated?

Before choosing an architecture, we need to decide how to evaluate it (lecture pp. 6–7).

```text
FAR = FP / (FP + TN)
FRR = FN / (FN + TP)
```

- **FAR**: the proportion of negative examples that produce false alarms.
- **FRR**: the proportion of genuine wake-word events that are missed.
- **FA/hour**: false activation events divided by hours of background audio.

Missing 5 out of 100 genuine triggers gives an FRR of 5%. Two false activations in 100 hours of background audio give 0.02 FA/hour. KWS evaluations often report FRR at a fixed false-alarm rate, such as **FRR @ 0.5 FA/hour**, making the trade-off explicit.

The denominators matter. FAR is a proportion over a defined set of negative examples; FA/hour is an event rate over continuous audio. Changing how many windows a recording is split into changes the number of negative windows. Several adjacent high-scoring windows may also represent one false activation. Continuous detection therefore needs a rule for merging triggers before counting events. Window-level accuracy cannot directly describe the wake-word experience.

Lowering a threshold usually reduces misses and increases false alarms; raising it does the reverse. Compare the full trade-off curve, or compare FRR at the same FA/hour, and record detection latency too. Waiting for future frames or several confirmation frames contributes to that latency.

A classic benchmark is Speech Commands:

| Version | Size               | Content                                     |
| ------- | ------------------ | ------------------------------------------- |
| v1      | About 65,000 clips | 30 short words, roughly one second per clip |
| v2      | 105,829 utterances | 35 words                                    |

Common words include “yes,” “no,” “stop,” and “go.” A common 12-class setup uses 10 target words, unknown, and silence. Unknown covers speech outside the target vocabulary; silence covers silence or background audio. They support different kinds of rejection.

**Short-word classification and continuous wake-word detection are related evaluation problems with different conditions.** A one-second classification clip usually supplies the boundaries. A device must also handle long background streams, word boundaries, repeated triggers, and speech immediately following the wake phrase. The lecture's model summary on p. 22 helps show changes in architecture and scale. Its accuracy figures should be read with their class definitions, data splits, and evaluation protocols rather than treated as a strict ranking across papers.

## From templates and HMMs to lightweight neural networks

### Templates and HMMs: handling timing variation

One early approach is template matching. Record “Xiao Ai Tong Xue,” save its acoustic features, and compare incoming audio with the template. A sufficiently close match triggers the system. Even the same person says the phrase differently each time: the first part might last 300 ms in the template and 500 ms in a new recording.

Dynamic time warping (DTW), discussed in the earlier ASR notes, allows local stretching of the time axis to find a correspondence between sequences. It addresses speaking-rate variation and can support recorded enrollment or custom phrases. A single template has difficulty covering different voices, accents, and noise conditions.

An HMM represents a keyword as an ordered sequence of states, which can correspond to phonemes or subphonetic states. Each state can last several frames, accommodating different speaking rates. In a traditional GMM-HMM, the GMM scores acoustic features against states, while the HMM constrains state order. A filler or background model explains sounds outside the keyword. Search runs over paths containing background and keyword states, identifying a segment with enough evidence for the keyword while explaining the rest as background.

### DNNs: local decisions and post-processing

An early DNN pipeline can be understood as:

```text
Neighboring acoustic frames → Fully connected network
                           → Frame-level label probabilities
                           → Smoothing and trigger logic
```

A single instant is rarely enough to identify “Hey Siri,” so the network receives nearby frames too. Predictions can fluctuate across time, motivating temporal smoothing. The network makes local acoustic decisions; post-processing combines the evidence over time. This resembles early ASR pipelines, where acoustic scores and temporal structure are handled by different components.

The lecture's example uses 30 past frames and 10 future frames. Those future frames require waiting, even if computation is fast. One way to smooth the posterior for label `i` is to average its most recent `w` frames:

```text
p_bar[i,j] = (∑_{k=h}^j p[i,k]) / (j − h + 1)
h = max(1, j − w + 1)
```

Here, `p[i,k]` is the posterior for label `i` at frame `k`. Smoothing still needs a threshold and temporal logic to produce an activation event.

The limitations are clear:

- Overlapping windows can repeat much of the same computation.
- Fully connected layers do not explicitly impose convolutional locality and weight sharing.
- Smoothing suppresses fluctuations but cannot flexibly learn which frames matter or their order.

### SVDF: factorizing time and feature weights

SVDF compresses DNNs through a structured factorization (p. 10). Suppose a window has `T` frames and each frame has `F` features. Ignoring bias, a hidden unit originally computes:

```text
a = σ(∑_t ∑_f W[t,f] x[t,f])
t = 1, …, T; f = 1, …, F
```

This requires a full `T × F` weight matrix. A rank-one approximation factors it into:

```text
W[t,f] ≈ α[t] β[f]
```

The computation becomes:

1. Project each frame's features: `z[t] = ∑_f β[f] x[t,f]`.
2. Combine across time: `a = σ(∑_t α[t] z[t])`.

For one rank-one filter, the weight count falls from `TF` to `T + F`. Higher ranks sum several components, increasing both capacity and cost.

**The change is to assume a factorization instead of learning an independent weight for every time-feature position.**

This suits streaming execution: cache historical `z[t]` values and compute projections only for new frames. The name relates to the low-rank idea of singular value decomposition; the slide spells it “Single Value Decomposition Filter.” I use the common abbreviation SVDF here.

The connection to ASR is bounded-context modeling, similar to TDNNs and temporal convolutions. SVDF caches a finite history, rather than recursively updating a hidden representation in the manner of an RNN.

### CNNs: reusing local patterns across positions

A DNN learns separate weights for different input positions. A CNN (Google, 2015, p. 11) instead uses:

```text
Time-frequency feature map → Local convolution
                           → Higher-level patterns → Classification
```

If a syllable arrives 100 ms later, the same kernel can still detect its local pattern. This introduces two assumptions:

- **Locality**: short temporal spans and local frequency regions contain useful patterns.
- **Weight sharing**: some patterns can use the same detector at different positions.

The lecture emphasizes that **fewer parameters do not necessarily mean less computation**. Shared kernels are still evaluated at many positions. A CNN can have fewer parameters than a DNN while requiring more FLOPs. This is also why ASR often uses CNNs as acoustic front ends: extract local structure before later components integrate broader information.

## Attention: learning which moments matter

### Attention pooling and ASR decoder attention

The early DNN makes frame-level predictions and smooths them manually. Attention-based KWS (Xiaomi, 2018, p. 12) first uses an RNN, GRU, LSTM, or CRNN to encode the input as `h₁, …, h_T`, then learns how to summarize it:

```text
eₜ = vᵀ tanh(Whₜ + b)
αₜ = exp(eₜ) / ∑_{j=1}^T exp(eⱼ)
c = ∑_{t=1}^T αₜ hₜ
```

`eₜ` scores a time position, `αₜ` is its normalized weight, and `c` goes to the classifier. If only the middle of a one-second recording contains the keyword, the model can assign that region more weight and reduce the influence of irrelevant background.

This attention performs **sequence summarization**. In an ASR attention-based encoder-decoder, attention selects acoustic information conditioned on the decoder state for each output token. The uses differ. Attention also does not establish streaming capability: normalization above covers the current `T`-frame input, and online execution depends on the encoder, window, and use of future context.

### Multihead attention: why constrain the heads?

A single context vector can lose information. Multiple heads produce `c₁, …, c_H` for classification, but they can also learn similar weights and repeatedly attend to the same region.

The work on pp. 13–15 encourages heads to extract different information through orthogonality constraints. Place normalized context vectors into the columns of `C`. The core of one regularizer is:

```text
L_inter ∝ ‖CᵀC − I_H‖²_F
```

Off-diagonal entries are inner products between heads. Reducing them discourages redundancy. The work also constrains attention score vectors across heads and encourages similar contexts from the same head across positive examples. Consistency within a head and diversity between heads are different objectives; every regularization term should not be interpreted as demanding greater orthogonality.

These constraints encourage particular behavior. They do not guarantee that each head corresponds to a phoneme or that adding more heads improves accuracy.

## Lighter architectures: time and frequency have different roles

### MatchboxNet: separable convolutions along time

The lecture's p. 16 makes a useful distinction: a spectrogram resembles an image, but its axes have different meanings. A local pattern occurring later can still be the same sound event. Moving it along frequency can change pitch, harmonics, or formant information. Image-style translation assumptions cannot be transferred unchanged to every frequency band.

MatchboxNet (NVIDIA, 2020, p. 17) follows QuartzNet's design, treating frequency features as input channels and using **1D time-channel separable convolutions**:

```text
Depthwise temporal convolution: extract time patterns per channel
                              ↓
Pointwise 1 × 1 convolution: mix information across channels
                              ↓
Residual blocks, later projections, and classification
```

Ignoring bias, a standard 1D convolution with temporal kernel width `K`, `C_in` input channels, and `C_out` output channels has `K × C_in × C_out` parameters. A depthwise temporal convolution followed by a pointwise convolution uses:

```text
K × C_in + C_in × C_out
```

Separating temporal filtering from channel mixing reduces parameters and multiply-accumulate operations. Actual speed still depends on operator support, memory access, and runtime implementation on the device.

### SubSpectral Normalization: separate frequency-band statistics

SSN (2021, p. 18) divides the frequency axis into sub-bands and applies BatchNorm-like normalization separately within them. For an input shaped `N × C × F × T`, `N` is the batch axis, `C` channels, `F` frequency, and `T` time.

Each sub-band uses its own statistics rather than directly pooling different frequency distributions together. This reflects the significance of frequency position. SSN is a normalization method that can accompany convolutions; it does not perform temporal alignment or keyword decisions itself.

### BC-ResNet: compute in one dimension and broadcast back

Broadcasted Residual Learning (2021, pp. 19–20) reduces repeated computation over the full time-frequency grid. After feature extraction with `f₂`, average pooling along frequency compresses the features to a temporal sequence. `f₁` models temporal variation, and its output is broadcast along frequency.

The lecture's BCResBlock is:

```text
y = x + f₂(x) + BC(f₁(AvgPool_F(f₂(x))))
```

BC denotes broadcasting. The temporal branch saves computation by operating on a compressed representation. Keeping the two-dimensional `f₂(x)` residual path prevents all frequency detail from being averaged away. SSN also appears in the illustrated block; the two mechanisms can work together to use frequency structure.

### Keyword Transformer: exchanging information between patches

Keyword Transformer (2021, p. 21) splits a Mel spectrogram into temporal patches, flattens and projects them into embeddings, adds positional information and a class token, and applies a Transformer encoder. The class token's representation is used for classification:

```text
Waveform → Mel spectrogram → Temporal patches
                           → Patch + Position embeddings
                                      ↓
                           Transformer encoder
                                      ↓
                             Class token → Class
```

Unlike final attention pooling, self-attention exchanges information between input patches. It can connect distant patches, but standard global attention has quadratic cost in token count. Short-word classification results alone do not establish suitability for always-on listening; window length, caching, and right context still matter.

The architectures address different questions:

| Architecture                       | Main change                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| DNN + smoothing                    | Make local frame-level decisions, then combine evidence through post-processing        |
| SVDF                               | Factor time-feature weights to reduce parameters                                       |
| CNN / MatchboxNet                  | Reuse local patterns and reduce cost through separable convolutions                    |
| Attention pooling / multiple heads | Learn which time positions to summarize and reduce redundancy between heads            |
| SSN / BC-ResNet                    | Use frequency-band structure and reduce computation over the full two-dimensional grid |
| Keyword Transformer                | Exchange information between patches through self-attention                            |

## Cascade systems: why not run the largest model continuously?

A more accurate model may consume too much power if it runs all the time. The Google and Apple examples on pp. 24–28 split detection across stages:

```text
Microphone
    ↓
Low-power processor: small always-on detector
    ↓ Only candidates advance
Application processor: more precise keyword check
    ↓
Speaker verification, when applicable
    ↓
Server verification / ASR → Understanding and execution
```

Google's examples include on-device KWS followed by server verification, and a mobile cascade of a small detector, a more accurate detector, and speaker verification. Apple's 2017 system also uses two passes: a low-compute detector followed by a larger DNN.

The 2023 Apple example is more specific. A DNN + HMM runs on the Always On Processor (AOP). Candidate audio reaches a high-precision checker on the Application Processor (AP), using a Conformer encoder with a CTC training objective and a decoder trained with cross-entropy. Personal devices also use speaker identification, followed by downstream ASR. These are historical designs cited by the lecture, rather than a claim that every device or current product uses the same pipeline.

This connects back to the ASR notes: CTC can supply a training signal for acoustic sequences, while the overall task remains deciding whether a valid trigger occurred. A trigger checker need not be a complete transcription service. Keyword verification asks what was said; speaker verification asks whether it came from the target user. They reject candidates for different reasons.

### Preserve candidates early and filter them later

A cascade spends expensive computation only on a small fraction of inputs. Suppose the first stage checks `r` times per second, each check costs `c₁`, its candidate pass rate is `q`, and each second-stage check costs `c₂`. A simplified average computation model is:

```text
Computation per second ≈ r(c₁ + q c₂)
```

Real cost also includes waking processors, transferring audio, and the duration of later processing. Passing too much background audio therefore consumes power even if every false candidate is eventually rejected.

A genuine trigger missed by the first stage never reaches the second. For two stages, the genuine-trigger acceptance probability is:

```text
P(pass₁, pass₂ | K)
  = P(pass₁ | K) × P(pass₂ | pass₁, K)
```

`K` denotes a genuine keyword event. The second factor is conditional on candidates that passed the first stage. Multiplying accuracies measured on independent test sets would not give this quantity. Early stages commonly favor retaining genuine candidates, while later stages improve precision. Thresholds must be tuned jointly, evaluating end-to-end FRR, FA/hour, latency, and power.

## Deployment: continuous and reliable execution

### Streaming caches: avoid recomputing the same history

Suppose a one-second window moves forward every 20 ms. Adjacent inputs overlap by 98%. Running the entire network for every window repeats much of the computation on old audio.

Streaming detection on pp. 30–31 reuses historical convolution results through caches:

```text
New audio / features arrive
            ↓
Read still-valid historical state at each layer
            ↓
Compute new positions → Update ring buffers → Discard expired state
            ↓
Update keyword score and trigger decision
```

Several layers may need caches. Kernel sizes, strides, dilation, and receptive fields determine the state each layer retains. Three issues should be distinguished:

- **Caching** reduces repeated computation but requires state memory.
- **Causality and right context** determine how much future audio must arrive.
- **Smoothing and trigger logic** determine when scores become an activation event.

Adding a buffer to an offline model does not establish low-latency execution. A convolution that accesses future frames still needs to wait. Normalization or pooling over a complete window also requires checking that online execution preserves the original output semantics. Rules for merging repeated triggers must agree with event-level evaluation.

### Transfer learning: what can a few examples achieve?

The few-shot system on p. 32 first learns an embedder from about 1.4 million examples covering 760 keywords in nine European languages. It then adapts a three-class classifier using five examples of a new target word and 128 non-target examples from a prepared bank. A common three-class setup uses target, unknown, and background.

The idea is to reuse an existing speech representation and learn a new classification boundary from a small amount of data. **“Only five examples” describes adaptation; substantial pretraining data came first.** Transfer to a new language, accent, or device still needs testing in that setting. Few-shot experimental results do not guarantee reliable detection of arbitrary custom wake words.

### Class imbalance: most audio is background

In continuous listening, most moments contain no wake word. A system that always predicts “no keyword” can have high window-level accuracy and be useless. The lecture's p. 33 lists three training strategies:

- **Oversampling**: include keyword examples more frequently in training batches.
- **Class-weighted cross-entropy**: give different classes different loss weights.
- **Focal loss**: reduce the contribution of easy examples so training focuses more on difficult ones.

For one example, focal loss can be written as:

```text
L_focal = −α_y (1 − p_y)^γ log(p_y)
```

`p_y` is the predicted probability of the correct class, `α_y` its class weight, and `γ ≥ 0` controls downweighting of easy examples. These methods change training signals. They do not replace testing on realistic background audio or ensure that a chosen threshold satisfies a target FA/hour.

### Robustness: hard negatives matter alongside positive examples

Page 34 lists multitask learning, augmentation, TTS-generated data, hard negative mining, federated learning, and data cleaning.

Augmentations include pitch and tempo changes, time shifts, resampling, room impulse response (RIR) simulation, and background noise. They expand coverage of voice and speaking-rate variation, keyword position, recording conditions, and acoustic environments. Resampling to simulate a recording pipeline should be distinguished from changing playback speed without preserving duration; the effects differ.

Hard negative mining collects audio that actually causes false triggers, such as similar-sounding phrases, television dialogue, or other people's speech, and uses it as negative training data. It adds examples near the decision boundary rather than simply more random noise. Recordings used for mining should be separate from the final test recordings, so fixing known examples is not mistaken for generalization.

KWS and ASR can also share representations and train jointly:

```text
L = λ L_KWS + (1 − λ) L_ASR,  0 ≤ λ ≤ 1
```

ASR provides finer supervision about speech content, while KWS keeps the detector aligned with its target task. Benefits depend on data, loss weights, and training. TTS can expand pronunciation and speaker coverage, but synthetic speech differs from microphone recordings. Federated learning changes how training uses data across devices, while cleaning improves recordings and labels. Neither automatically solves false activations.

### Optimization: parameters, computation, and power are distinct

The final slide lists complementary approaches:

| Method                                  | Main purpose                                                                                                     | What to verify                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Graph optimization / reparameterization | Fuse compatible operators or convert training structures into simpler inference graphs; the lecture cites RepCNN | Output after conversion and execution time on the target hardware            |
| Quantization                            | Reduce weight and activation precision, using quantization-aware training (QAT) when needed                      | Score and threshold changes, accuracy, latency, and memory                   |
| Pruning                                 | Remove weights or structures; structured pruning maps more directly to smaller computation graphs                | Whether removed components actually reduce device computation                |
| Parameter sharing / adapters            | Reuse parameters and reduce added parameters or adaptation cost                                                  | Whether the saving concerns trainable parameters, storage, or inference work |
| Distillation                            | Use a teacher to guide a student while retaining detection capability                                            | The student's event-level performance within the same deployment budget      |

After quantization or pruning, reselect the operating threshold and evaluate the full cascade. Fewer parameters generally help storage, while lower MAC counts mean fewer theoretical multiply-accumulate operations. Whether always-on listening saves power still requires measurements of execution frequency, cache memory, processor wake-ups, and data movement on the target device.

## Lecture and further reading

- Main source: [Lecture 4 — Keyword Spotting](https://github.com/severilov/DL-Audio-AIMasters-Course/blob/main/lectures/lecture04_DLAudio2026.pdf). Task and evaluation: pp. 4–7; models: pp. 9–22; cascades: pp. 24–28; practice: pp. 30–35.
- DNN / SVDF / CNN: Chen et al., _Small-footprint keyword spotting using deep neural networks_ (2014); Nakkiran et al., _Compressing deep neural networks using a rank-constrained topology_ (2015); Sainath et al., _Convolutional neural networks for small-footprint keyword spotting_ (2015).
- Attention and efficient architectures: Shan et al. (2018); Lee et al., _Orthogonality Constrained Multi-Head Attention for Keyword Spotting_ (2019); _MatchboxNet_ (2020); _SubSpectral Normalization_, _Broadcasted Residual Learning_, and _Keyword Transformer_ (all 2021). Full titles and publication details appear on the corresponding slides.
- Systems and practice: _A Cascade Architecture for Keyword Spotting on Mobile Devices_ (2017); _Voice Trigger System for Siri_ (2023); _Streaming keyword spotting on mobile devices_ (2020); _Few-Shot Keyword Spotting in Any Language_ (2021).
