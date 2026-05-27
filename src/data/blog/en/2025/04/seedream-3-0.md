---
author: Yuhao Chen
pubDatetime: 2025-04-21T05:00:00.000Z
modDatetime: 2025-04-22T02:17:54.000Z
title: "Seedream 3.0 Technical Report"
slug: seedream-3-0
featured: false
draft: false
tags:
  - Technology
  - Doubao
  - Paper
description: "Reading notes on the Seedream 3.0 technical report"
---

Original link: [https://arxiv.org/pdf/2504.11346](https://arxiv.org/pdf/2504.11346)

## Introduction

Seedream 2.0 was already very good, but it still had some problems: the model's alignment on complex prompts still needed improvement, especially in cases involving numerical precision and spatial relationships among multiple objects; 2.0's ability to generate text inside images still needed improvement; there were issues with image aesthetics; and there were also image clarity problems. For these issues, Doubao made several improvements: at the data level, it introduced twice as much high-quality data; it added training steps and techniques, such as mixed resolution training, multimodal ROPE, a new representation alignment loss, and resolution-aware sampling. Finally, it also improved post-training and generation acceleration. In short, 3.0 is an incremental change over 2.0, but only one month had passed. From this, you can see the real quality of the ByteDance Doubao team, and the importance of good AI infra for continuous research and product iteration.

## Data

The paper mentions that in the 2.0 stage, they used a strict data filtering mechanism, which limited the amount of training data. In 3.0, Doubao uses a new filtering mechanism that keeps data with defects under 20%, and during training uses a spatial attention mask so these regions are excluded from training. This successfully expands the training data by almost 20% while maintaining model stability.

## Pre-training

The model architecture follows 2.0, only increasing the training parameters and adding the following techniques:  
1. Mixed resolution training. Specifically, because transformers naturally support sequences of different lengths, the Doubao team first uses 256^2 data for pre-training, then fine-tunes on higher-resolution data, from 512^2 to 2048^2. It also adds size embedding as an extra condition, probably using cross attention, so the model can still perform well under unseen resolutions.  
2. Cross-modality ROPE. In 2.0, they used scaling ROPE. In 3.0, this technique is improved. In the past, we would apply 1D ROPE to text and 2D ROPE to images. But in CM-ROPE, text is also treated as one-dimensional 2D, applying 2D ROPE and projecting it into 2D space to associate with images.  
3. It uses a flow matching loss function and adds alignment loss, used to align its own MMDiT with DINOv2, which can help the acceleration model converge.  
4. Resolution-aware Timestep Sampling is a diffusion model training technique. The principle is to change the distribution we sample from under different resolutions: high-resolution images make the sampling distribution lean more toward lower SNRs and higher noise levels. During training, it uses the dataset's average resolution. During inference, it uses the expected resolution to determine the shift factor. The concrete method is to first sample from a log-normal distribution, then shift according to the calculated shift factor.

## Post-training

Compared with 2.0, 3.0 removes the refiner stage because the model itself can already generate images at different resolutions. Besides that, it made the following improvements: it trained more captioning models for the CT and SFT stages, allowing the model to better understand aesthetics, style, and layout in prompts; and it balanced the amount of data across different resolutions.

Another point is that it uses a VLM rather than CLIP as the reward function. The concrete method is as follows:  
1. Instruction as Query: The model receives a prompt, such as "A cat sitting on a couch."  
2. Formulating the Question: This prompt is transformed into a question like, "Does this image depict a cat sitting on a couch? Please answer Yes or No."  
3. Evaluating with VLM: The VLM processes the generated image and the question, outputting probabilities for "Yes" and "No."  
4. Deriving the Reward: The probability assigned to "Yes" is normalized and used as the reward signal. A higher probability indicates better alignment between the image and the prompt.

### Model acceleration

Seedream 3.0's model acceleration is based on Hyper-SD and RayFlow. Compared with traditional diffusion models, where all samples follow the same Gaussian distribution path during denoising, Seedream implements personalized single paths for different samples, improving model stability and generation diversity. It also uses a pre-trained model to estimate noise. This method allows the model to converge with maximum likelihood during the noise-adding and denoising process, so the model can now obtain very good results with fewer steps. For training acceleration, they trained a neural net combined with Stochastic Stein Discrepancy (SSD) to predict which timestamp will produce the largest training loss. Therefore, during training sampling, unlike traditional uniform sampling, it can more efficiently sample the most important timestamps. Combining the above work, Doubao's model can achieve, at higher efficiency, the effect that ordinary diffusion models need 50 sampling steps to reach.
