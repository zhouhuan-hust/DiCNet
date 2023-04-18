# DiCNet
This repository provides the PyTorch implementation of the paper: [Anomaly Discovery in Semantic Segmentation via Distillation Comparison Networks](https://arxiv.org/abs/2112.09908)

## Introduction
Our key observation is that semantic classification plays a critical role in existing approaches, while the incorrectly classified pixels are easily regarded as anomalies. Such a phenomenon frequently appears and is rarely discussed, which significantly reduces the performance of anomaly discovery. To this end, we propose a novel Distillation Comparison Network (DiCNet). It comprises of a teacher branch which is a semantic segmentation network that removed the semantic classification head, and a student branch that is distilled from the teacher branch through a distribution distillation. We show that the distillation guarantees the semantic features of the two branches hold consistency in the known classes, while reflect inconsistency in the unknown class. Therefore, we leverage the semantic feature discrepancy between the two branches to discover the anomalies. DiCNet abandons the semantic classification head in the inference process, and hence significantly alleviates the issue caused by incorrect semantic classification.

## How to use
### Environment
* Python 3.8
* Pytorch 1.10
### Install
#### Create a virtual environment and activate it
```
conda create -n dicnet python=3.8
conda activate dicnet
```
#### Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python
pip install numpy
pip install tqdm
pip install matplotlib
pip install easydict
pip install scipy
pip install scikit-learn
```

### Data Preparation
Download [StreetHazards train](https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar), [StreetHazards test](https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar)
<br>Put them in data/

### Train
Use [this repository](https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/5c2e9f6f3a231ae9ea150a0019d161fe2896efcf)  to train a pspnet as the teacher network
 <br>Train the student network
```
python train_student.py
```

### Test
```
python evaluation_with_seg.py
```

### Pretrained Model
[DiCNet](https://drive.google.com/drive/folders/1H30m7ZeU4JOtjVUUAv5DFn8GGo4rWtrB?usp=sharing)

## Results on StreetHazards dataset

| Method | AUPR(%) | FPR95(%) | AUROC(%) | 
|---|---|---|---|
| MSP | 6.6 | 33.7 | 87.7 |
| Deep Ensemble | 7.2 | 25.4 | 90.0 | 
| TRADI | 7.2 | 25.3 | 89.2 | 
| SynthCP |  9.3 | 28.4 | 88.5 | 
| PAnS |  8.8 | 23.2 | 91.1 | 
| DiCNet | 16.2 | 17.6 | 93.4 | 

## Citation

If you find this useful in your research, please consider citing:

    @article{zhou2021anomaly,
      title={Anomaly Discovery in Semantic Segmentation via Distillation Comparison Networks},
      author={Zhou, Huan and Gong, Shi and Zhou, Yu and Zheng, Zengqiang and Liu, Ronghua and Bai, Xiang},
      journal={arXiv preprint arXiv:2112.09908},
      year={2021}
    }

