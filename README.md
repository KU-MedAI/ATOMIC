
<h1 align="center">ATOMIC: A graph attention neural network for ATOpic dermatitis prediction on human gut MICrobiome</h1>
<p align="center">
    <a href="https://journals.plos.org/ploscompbiol/"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ploscompbiol%2725&color=blue"> </a>
    <a href="https://github.com/KU-MedAI/ATOMIC"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
</p>

![The proposed model](img/overview.png)

## Abstract
Atopic dermatitis (AD) is a chronic inflammatory skin disease driven by complex interactions among genetic, environmental, and microbial factors, yet its etiology remains incompletely understood. Recent studies have reported the role of gut microbiota dysbiosis in AD pathogenesis, leading to an increased interest in microbiome-targeted therapeutic strategies such as probiotics and fecal microbiota transplantation (FMT). Building on these findings, recent advances in computational modeling have introduced machine learning and deep learning-based approaches to capture the nonlinear relationships between gut microbiota and disease. These models, however, focus on diseases other than AD and often fail to capture complex microbial interactions or incorporate microbial genomic information. To address these limitations, we propose ATOMIC, an interpretable graph attention network-based model that integrates microbial co-expression networks to predict AD. To train and test our model, we collected and processed 99 gut microbiome samples from adult patients with AD and healthy controls at Kangwon National University Hospital (KNUH). The network incorporates microbial genomic information as node features, enhancing its ability to capture functionally relevant microbial patterns. As a result, ATOMIC achieved an AUROC of 0.810 and an AUPRC of 0.927 on the KNUH dataset. Furthermore, ATOMIC identified microbes potentially associated with AD prediction and proposed candidate microbial biomarkers that may inform future therapeutic strategies. To facilitate future research, we publicly released the gut microbial abundance dataset from KNUH.

### Installation and data preparation
Our code is based on the following libraries:

```
torch==2.3.0+cu118
torch-geometric==2.6.1
torch_scatter==2.1.2
scikit-learn==1.5.2
```

The microbial co-expression data used in the paper can be obtained following these [instructions](./datasets/README.md).

### Usage

Simply run 
```
bash ./scripts/train.sh
```
Run with a fixed epochs
```
bash ./scripts/train_fixed_epoch.sh
```
using both the train set and valid set for training

### Contact
If you have any question regard our study, please contact us (**mjjeon@korea.ac.kr**)

### Funding
This study was supported by the MSIT(Ministry of Science and ICT), Korea, under the Technology development Program(S3364091) of MSS, ICAN (ICT Challenge and Advanced Network of HRD) program (IITP-2025-RS2022-00156439) supervised by the IITP (Institute of Information & Communications Technology Planning & Evaluation) and Bio&Medical Technology Development Program of the National Research Foundation (NRF) funded by the Korean government (MSIT) (No. RS-2024-00441029).
----------

If you find our paper and repo useful, please cite our paper: -->

```bibtex -->
temp
```

Ack: The readme is inspired by [CIGA](https://github.com/LFhase/CIGA).
