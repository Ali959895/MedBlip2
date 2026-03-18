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
## MedBLIP2 architecture
<p align="center">
  <img src="image/Diagram.png" alt="MedBLIP2 architecture" width="800">
</p>
Figure 1. The overall architecture and workflow of our proposed vision-language model for medical image analysis.

## MedQFormer architecture
<p align="center">
  <img src="image/MedQFormer.png" alt="MedBLIP2 architecture" width="800">
</p>
Figure 2. MedQFormer queries the frozen image encoder’s output
embeddings to extract compact visual representations, which
are converted into soft prompt tokens and injected into the frozen
LLM to enable instruction-guided medical inference.
## Comparison with State-of-the-Art
<p align="center">
  <img src="image/metrics.JPG" alt="metrics" width="800">
</p>
Table 1. Performance comparison across multiple evaluation metrics
for gastrointestinal tract analysis.
## MedBLIP2 evaluation on the LC25000 dataset
<p align="center">
  <img src="image/LC2500.JPG" alt="LC25000" width="800">
</p>
Table 2. Predicted labels on LC25000 dataset. Each column represents
the same representative histopathology image patch, while each row reports the prediction produced by different models.
## MedBLIP2 evaluation on the Kvasir  dataset
<p align="center">
  <img src="image/Kvasir.JPG" alt="Kvasir" width="800">
</p>
Table 3. Predicted labels for MedBLIP2 (2.7B, 6.7B) and baseline models compared to the ground-truth class (dyed-lifted-polyps, dyeless-
polyps, normal-cecum, normal-pylorus, normal-z-line, polyps, ulcerative-colitis, and esophagitis).
## MedBLIP2 evaluation on the HAM10000  dataset
<p align="center">
  <img src="image/HAM10000.JPG" alt="HAM10000" width="800">
</p>
Table 4. Qualitative summary of predicted labels for representative HAM10000 samples. The reference row lists the ground-truth classes.
Correct predictions are marked with a green check, while misclassified samples are represented by the incorrectly predicted class followed
by a red cross.

## Installation
git clone https://anonymous.4open.science/status/MedBlip2-F370.git  

cd MedBlip2  

pip install -r requirements.txt
## Datasets
- LC25000  (https://www.kaggle.com/datasets/javaidahmadwani/lc25000)
- Kvasir   (https://datasets.simula.no/kvasir)
- HAM10000 (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Training and Evaluation
bash scripts/train_lc25000.sh  

bash scripts/train_kvasir.sh  

bash scripts/train_ham10000.sh
## Citation
@article{medblip2,
  title={Medical Vision--Language Models for Robust Disease Diagnosis},
  author={Anonymous},
  year={2026}
}

