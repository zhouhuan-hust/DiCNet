# DiCNet
This repository provides the PyTorch implementation of the paper: [Anomaly Discovery in Semantic Segmentation via Distillation Comparison Networks](https://arxiv.org/abs/2112.09908)
## Introduction
Our key observation is that semantic classification plays a critical role in existing approaches, while the incorrectly classified pixels are easily regarded as anomalies. Such a phenomenon frequently appears and is rarely discussed, which significantly reduces the performance of anomaly discovery. To this end, we propose a novel Distillation Comparison Network (DiCNet). It comprises of a teacher branch which is a semantic segmentation network that removed the semantic classification head, and a student branch that is distilled from the teacher branch through a distribution distillation. We show that the distillation guarantees the semantic features of the two branches hold consistency in the known classes, while reflect inconsistency in the unknown class. Therefore, we leverage the semantic feature discrepancy between the two branches to discover the anomalies. DiCNet abandons the semantic classification head in the inference process, and hence significantly alleviates the issue caused by incorrect semantic classification.
## How to use
### Environment
* Python 3.8
* Pytorch 1.10
