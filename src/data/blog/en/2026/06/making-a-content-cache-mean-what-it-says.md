---
author: Yuhao Chen
pubDatetime: 2026-06-23T23:25:00.000Z
title: "Making a content cache mean what it says"
slug: making-a-content-cache-mean-what-it-says
featured: false
draft: false
tags:
  - Technology
  - TTS
  - MLSys
description: "Investigating a subtle batch-invariance bug in MOSS-TTS Local v1.5's reference-audio cache, where content-addressed keys hid execution-shape-dependent values."
---

MOSS-TTS Local v1.5 is not served like a normal language model.

A single request does not just enter an autoregressive decode loop and exit with tokens. It moves through a reference-audio encoder, an AR backbone, a frame-local audio-token sampling path, and a streaming vocoder. That makes the serving problem more interesting than "make decode faster." Each stage has its own correctness contract, and each optimization can quietly change the semantics of the system.

My work started from one of those quiet contracts: the reference-audio cache.

The cache looked simple. For a given reference audio file, compute a content hash, encode the audio once, and reuse the resulting codec tokens later.

```text
same audio bytes -> same cache key -> same reference codes
```

That is the contract a content-addressed cache appears to promise. If the same speaker reference is reused across many requests, caching is valuable: reference encoding is not free, while a cache hit is essentially just a memory lookup. So the first instinct is to preserve batching on cache miss, make the first fill fast, and then let later requests hit the cache.

But I was not convinced the cache value was actually canonical.

The suspicious part was the interaction between caching and batching. A cache miss could be filled through a batched reference encoder. That means the first request for an audio file might encode it alone, or it might encode it next to other reference audios, depending on request timing. If those two executions produce the same tokens, the design is fine. If they produce different tokens, the cache becomes order-dependent.

So the first question became very small:

```text
Does encode([A])[0] equal encode([A, B])[0]?
```

The answer was no.

Repeated solo encodes were stable. Repeated batched encodes were also stable. This was not ordinary run-to-run nondeterminism. The model was deterministic for a fixed execution shape. The problem was batch invariance: the same audio could produce different discrete codec tokens when encoded alone versus as part of a batch.

That distinction mattered. If the output were random across repeated runs, the fix would point toward seed control, deterministic kernels, or removing nondeterministic operations. But here the system was more subtle. It was deterministic from the server's full-input perspective, but nondeterministic from an individual request's perspective, because the concurrent neighbors of that request changed the batch shape.

The next step was to avoid guessing.

A tempting explanation was padding leakage: maybe the second audio changed padding length, attention masks, or valid-region handling. I tested that by controlling length and padding content. Same-length pairs still produced mismatches, so this was not only a variable-length artifact. Then I changed the padding content: zero padding, random padding, repeat-last padding. The valid-region tokens stayed the same across those padding variants. That made attention-mask leakage less likely.

The remaining hypothesis was numeric shape sensitivity.

To test that, I traced the codec encoder layer by layer. The first concrete divergence appeared at `encoder.7.input_proj`. The input to the linear layer was identical, but the output differed by a small BF16-scale amount. I then isolated the real saved input and real saved weights and reproduced the same difference with a standalone linear call.

That was the important point. The drift was not caused by high-level transformer logic. It was a shape-sensitive GEMM effect. Small BF16 differences are usually harmless in continuous networks, but this path ends in discrete audio-code quantization. A small activation change can cross a quantization boundary and become a different token ID.

At that point, the cache bug became clear.

The problem was not that MOSS-TTS had a cache. The cache was useful and should stay. The problem was that a content-only key was being paired with a non-canonical value.

```text
file:{content_hash} -> whichever batch shape filled the cache first
```

That is not a content cache. It is a race between serving schedules.

There were three possible designs.

The first option was to keep the content key but make the value canonical. On cache miss, encode the reference audio through a true solo path and store that result. Batched uncached execution can still exist elsewhere, but the cache fill has one stable meaning.

```text
file:{content_hash} -> canonical B=1 reference-audio codes
```

The second option was to include execution shape in the key. That preserves batched miss-fill semantics, but it fragments the cache and is hard to make complete. Shape, dtype, device, layout, backend, CUDA version, PyTorch version, and GEMM algorithm selection can all matter. A key like that stops being a clean content-addressed cache.

The third option was to disable caching for batched paths. That is simple, but it gives up too much of the benefit under real serving workloads, especially when the same small set of reference voices is reused many times.

I preferred the first option: canonical content cache with true solo miss fill.

The reason is workload-driven. If reference voices are reused heavily, the dominant variable is cache hit rate, not miss-fill throughput. For example, with 10 reference voices and 10,000 requests, the important thing is to pay the reference encoder cost roughly 10 times, not 10,000 times. Whether those 10 first fills were batched is much less important than making sure each content key has one stable value.

The experiment changed the design question. I was no longer asking, "Can we batch reference encoding faster?" I was asking, "What must be true for this cache to be semantically correct?"

That is the broader lesson from this work. In TTS serving, performance optimizations often cross stage boundaries: batching, caching, streaming, CUDA graphs, and scheduling all interact. A local optimization can accidentally change the user-visible behavior of the system.

The fix is not to avoid optimization. The fix is to write down the contract first, then design experiments that try to break it.

For this cache, the contract should be simple:

```text
same audio content -> same canonical codec tokens
```

Once that is true, the rest of the serving system becomes easier to reason about. Cache hit and miss parity can be tested. CI failures become interpretable. Request ordering no longer decides the cached value. And the cache once again means what its name says: content-addressed.

Read the full article on [LMSYS](https://www.lmsys.org/blog/2026-06-17-moss-tts-local-v15).
