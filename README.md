# Medical Vision–Language Models for Robust Disease Diagnosis
This paper has been submitted for publication in PHAROS AI Factory for Medical Imaging & Healthcare (PHAROS-AIF-MIH) in conjunction with the IEEE Computer Vision and Pattern Recognition Conference (CVPR), 2026

## Abstract

Medicine inherently involves integrating diverse data modalities, making multimodal learning a crucial component of computer-aided diagnosis. Recent advances in generative vision-language models have opened new possibilities for medical applications but remain limited by domain adaptation and computational complexity. In this study, we propose MedBLIP2 (2.7B), a lightweight vision-language pretraining framework tailored for the medical domain. The model leverages frozen off-the-shelf image encoders and large language models, linked through a novel MedQFormer module that effectively aligns visual and textual representations. Based on the BLIP-2 architecture, MedBLIP2 (2.7B) employs a two-stage pretraining strategy: (1) vision-language representation learning from a frozen image encoder and (2) vision-to-language generative learning from a frozen language model. Experimental results demonstrate that MedBLIP2 (2.7B) achieves state-of-the-art performance across multiple medical classification applications, surpassing BLIP-base and CLIP by 40.7\% and 23.1\%, respectively, on the lung tissue identification task. Further evaluations across diverse datasets confirm the model’s robust generalization and strong potential to enhance AI-assisted clinical diagnostics.

In this work:
1. We propose MedBLIP2 (2.7B/6.7B), a lightweight medical vision–language framework that integrates heterogeneous base models with a specialized BERT-based vision encoder (Q-Former) for high-fidelity feature extraction and robust multimodal representation learning. Pretrained on paired medical image–text data, it excels at classification tasks as a versatile CAD system extensible to new modalities and pathologies.

2. We introduce MedQFormer, which extracts medical image features and aligns them with text for seamless fusion into the LM's embedding space. This bridges heterogeneous medical data modalities, enabling versatile clinical applications.

3. We propose a unified adaptation pipeline that fully fine-tunes the Q-Former to learn domain-specific medical features and mitigate distribution shift across modalities, while systematically tuning activations and optimizers for improved convergence, efficiency, and generalization on downstream medical VLM classification tasks.


