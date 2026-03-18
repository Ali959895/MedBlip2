# Medical Vision–Language Models for Robust Disease Diagnosis
This paper has been submitted for publication in PHAROS AI Factory for Medical Imaging & Healthcare (PHAROS-AIF-MIH) in conjunction with the IEEE Computer Vision and Pattern Recognition Conference (CVPR), 2026

## Overview
**MedBLIP2** is a lightweight medical vision-language framework built on BLIP-2 for robust disease diagnosis across multiple medical imaging domains.

This repository provides:
- **MedQFormer-based adaptation**
- **Benchmarking on LC25000, Kvasir, and HAM10000**
- **Training, evaluation, and visualization scripts**
- **Comparisons with BLIP, CLIP, and ResNet baselines**
  
## Highlights
- **Medical-domain adaptation** of BLIP-2 for disease classification
- **Unified benchmark** across histopathology, GI endoscopy, and dermoscopy
- **Strong performance** with MedBLIP2-OPT-2.7B and MedBLIP2-OPT-6.7B
- **Reproducible experimental pipeline**

## Abstract

Medicine inherently involves integrating diverse data modalities, making multimodal learning a crucial component of computer-aided diagnosis. Recent advances in generative vision-language models have opened new possibilities for medical applications but remain limited by domain adaptation and computational complexity. In this study, we propose MedBLIP2 (2.7B), a lightweight vision-language pretraining framework tailored for the medical domain. The model leverages frozen off-the-shelf image encoders and large language models, linked through a novel MedQFormer module that effectively aligns visual and textual representations. Based on the BLIP-2 architecture, MedBLIP2 (2.7B) employs a two-stage pretraining strategy: (1) vision-language representation learning from a frozen image encoder and (2) vision-to-language generative learning from a frozen language model. Experimental results demonstrate that MedBLIP2 (2.7B) achieves state-of-the-art performance across multiple medical classification applications, surpassing BLIP-base and CLIP by 40.7\% and 23.1\%, respectively, on the lung tissue identification task. Further evaluations across diverse datasets confirm the model’s robust generalization and strong potential to enhance AI-assisted clinical diagnostics.

## Tools and Libraries
- PYTHON
- PYTORCH
- NUMPY
- PANDAS

# Figures



