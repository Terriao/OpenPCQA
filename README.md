<div align="center">

<img src="openpcqa.png" alt="OpenPCQA" width="640"/>

# OpenPCQA

**An Open-Source Algorithm Library of Point Cloud Quality Assessment Based on Deep Learning**

[![License](https://img.shields.io/badge/license-Academic-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-✓-EE4C2C?logo=pytorch&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-✓-FF6F00?logo=tensorflow&logoColor=white)](#)
[![MindSpore](https://img.shields.io/badge/MindSpore-✓-0072C6)](#)
[![Mirror](https://img.shields.io/badge/mirror-OpenI-success)](https://openi.pcl.ac.cn/OpenPCQA/OpenPCQA)

*A unified multi-framework benchmark for deep-learning-based PCQA methods.*

</div>

---

## 📖 Overview

**OpenPCQA** collects representative deep-learning methods on Point Cloud Quality Assessment (PCQA), provides reproducible source code in **MindSpore**, **PyTorch**, and **TensorFlow**, and benchmarks their performance under a unified protocol.

The goal of this repository is to lower the entry barrier for PCQA research by offering:

- ✅ **Unified implementations** of 5 representative PCQA algorithms
- ✅ **Cross-framework parity** — every method runs on PyTorch, TensorFlow, and MindSpore
- ✅ **Benchmark results** on SJTU-PCQA, WPC, LS-PCQA, and PRLD datasets
- ✅ **Reproducibility** — configs, seeds, and pretrained weights included

> 🔗 **Mirror repository (OpenI):** <https://openi.pcl.ac.cn/OpenPCQA/OpenPCQA>

---

## 📑 Table of Contents

- [Algorithm Zoo](#-algorithm-zoo)
  - [1.1 VQA-PC](#11-vqa-pc) — projection-based, NR
  - [1.2 PRL-GQA](#12-prl-gqa) — model-based, NR, geometry-only
  - [1.3 IT-PCQA](#13-it-pcqa) — projection-based, NR, domain adaptation
  - [1.4 PQA-Net](#14-pqa-net) — projection-based, NR
  - [1.5 ResSCNN](#15-resscnn) — model-based, NR, sparse-conv
- [Contributors](#-contributors)
- [Contact](#-contact)

---

## 🧠 Algorithm Zoo

| # | Method | Year | Type | Input | Datasets |
|---|--------|------|------|-------|----------|
| 1.1 | [**VQA-PC**](#11-vqa-pc) | 2022 | Projection (video) | Rendered videos | SJTU, WPC, LS-PCQA |
| 1.2 | [**PRL-GQA**](#12-prl-gqa) | 2022 | Model-based (rank) | Raw point cloud | PRLD |
| 1.3 | [**IT-PCQA**](#13-it-pcqa) | 2022 (CVPR) | Projection (DA) | Projected images | SJTU, TID2013 |
| 1.4 | [**PQA-Net**](#14-pqa-net) | 2021 (TCSVT) | Projection (multi-view) | Projected images | distortion.zip |
| 1.5 | [**ResSCNN**](#15-resscnn) | 2023 (ACM TOMM) | Model-based (sparse) | Raw point cloud | LS-PCQA |

> All methods are **no-reference (NR)**.

---

### 1.1 VQA-PC

> 📅 **2022 Sep 11** · 🏷️ Projection-based · 🎯 No-reference
> 🔗 Subproject: <https://openi.pcl.ac.cn/OpenPCQA/VQA_PC>

<p align="center"><img src="VQA-PC.png" alt="VQA-PC architecture" width="720"/></p>
<p align="center"><em>Figure 1 · Network structure of VQA-PC</em></p>

**Idea.** Treats PCQA as a video quality assessment (VQA) problem by rendering the point cloud as a video captured by a moving camera, then extracting spatial and temporal quality-aware features via a trainable 2D-CNN and a pretrained 3D-CNN respectively.

**Keywords:** point cloud quality assessment · moving camera videos · no-reference · VQA

#### 📂 File structure

```
root/
├── mindspore/         # MindSpore code + pretrained models
├── tensorflow/        # TensorFlow code + pretrained models
├── pytorch/           # see https://github.com/zzc-1998/VQA_PC
└── paper.pdf          # original paper
```

#### ⚙️ Environment

**MindSpore**
- Ubuntu 16.04 · CUDA 11.1.105 · Python 3.7.11
- `mindspore==2.0.0.dev20230109` ([install guide](https://www.mindspore.cn/install))

**TensorFlow**
- Ubuntu 16.04 · CUDA 10.1.243 · Python 3.7.6
- `tensorflow-gpu==2.3.1`

**Datasets**
- [WPC](https://github.com/qdushl/Waterloo-Point-Cloud-Database)
- [LS-PCQA](https://smt.sjtu.edu.cn/database/large-scale-point-cloud-quality-assessment-dataset-ls-pcqa/)
- [SJTU](https://smt.sjtu.edu.cn/database/point-cloud-subjective-assessment-database/)

#### ▶️ Run

```bash
# MindSpore
cd mindspore

# TensorFlow
cd TensorFlow
python ./train/train_SJTU.py    # train
python ./test/test.py           # test
```

#### 📊 Benchmark

**Table 1 · SJTU dataset**

| Source | SRCC | PLCC | KRCC | RMSE |
|:--|:-:|:-:|:-:|:-:|
| Paper | 0.8509 | 0.8635 | 0.6585 | 1.1334 |
| PyTorch | 0.9125 | 0.9341 | 0.7634 | 0.8364 |
| **MindSpore** | **0.9136** | **0.9346** | 0.7619 | **0.8350** |
| TensorFlow | 0.8774 | 0.9002 | 0.7048 | 1.0253 |

**Table 2 · WPC dataset**

| Source | SRCC | PLCC | KRCC | RMSE |
|:--|:-:|:-:|:-:|:-:|
| Paper | 0.7968 | 0.7976 | 0.6115 | 13.6219 |
| PyTorch | 0.8173 | 0.8226 | 0.6310 | 12.9618 |
| MindSpore | 0.8069 | 0.8085 | 0.6188 | 13.4193 |
| **TensorFlow** | **0.8296** | **0.8313** | **0.6457** | **12.6353** |

**Table 3 · Test time & GPU memory (Tesla T4)**

| Framework | Test time (s) | GPU memory (MB) |
|:--|:-:|:-:|
| PyTorch | 29.937 | **1358** |
| **MindSpore** | **26.405** | 3110 |
| TensorFlow | 44.887 | 4732 |

> The three frameworks yield similar accuracy. MindSpore leads on SJTU; TensorFlow leads on WPC. MindSpore is fastest at inference; PyTorch uses the least GPU memory.

#### 📚 Citation

```bibtex
@article{zhang2022treating,
  title  = {Treating Point Cloud as Moving Camera Videos: A No-Reference Quality Assessment Metric},
  author = {Zhang, Zicheng and Sun, Wei and Zhu, Yucheng and Min, Xiongkuo and Wu, Wei and Chen, Ying and Zhai, Guangtao},
  journal= {arXiv preprint arXiv:2208.14085},
  year   = {2022}
}
```

**Maintainer:** Ye Hua · `yeh@pcl.ac.cn`

---

### 1.2 PRL-GQA

> 📅 **2022 Nov 2** · 🏷️ Model-based · 🎯 No-reference · 📐 Geometry-only
> 🔗 Subproject: <https://openi.pcl.ac.cn/OpenPCQA/PRL-GQA>

<p align="center"><img src="PRL-GQA.png" alt="PRL-GQA architecture" width="720"/></p>
<p align="center"><em>Figure 2 · Network structure of PRL-GQA</em></p>

**Idea.** The first pairwise learning framework for no-reference geometry-only PCQA. Takes a pair of point clouds as input and outputs their relative rank order.

**Keywords:** point cloud · geometry quality assessment · rank learning · NR PCQA

#### 📂 File structure

```
root/
├── MindSpore/         # MindSpore code + pretrained models
├── TensorFlow/        # TensorFlow code + pretrained models
├── PyTorch/           # weights only; source: https://zhiyongsu.github.io/Project/PRLGQA.html
└── paper.pdf          # original paper
```

#### ⚙️ Environment

Same as VQA-PC (MindSpore 2.0.0.dev20230109 / TensorFlow-gpu 2.3.1).

**Dataset:** [PRLD](https://zhiyongsu.github.io/Project/PRLGQA.html)

#### ▶️ Run

```bash
# MindSpore
cd MindSpore

# TensorFlow
cd TensorFlow

# train & test
python ./train_test.py
```

#### 📊 Benchmark — PRLD dataset

| Source | Accuracy | Test time (s) | GPU memory (MB) |
|:--|:-:|:-:|:-:|
| Paper | 0.9449 | — | — |
| **PyTorch** | **0.9260** | **624.1** | **2198** |
| MindSpore | 0.9159 | 1308.1 | 4012 |
| TensorFlow | 0.8924 | 1962.8 | 2692 |

> PyTorch wins on every dimension here: best accuracy, fastest inference, lowest GPU footprint. MindSpore uses ~3× more GPU memory than PyTorch.

#### 📚 Citation

```bibtex
@article{su2022no,
  title  = {No-reference Point Cloud Geometry Quality Assessment Based on Pairwise Rank Learning},
  author = {Su, Zhiyong and Chu, Chao and Chen, Long and Li, Yong and Li, Weiqing},
  journal= {arXiv preprint arXiv:2211.01205},
  year   = {2022}
}
```

**Maintainers:** Ye Hua · `yeh@pcl.ac.cn` · Gao Wenxu · `gaowx@stu.pku.edu.cn`

---

### 1.3 IT-PCQA

> 📅 **2022 CVPR** · 🏷️ Projection-based · 🎯 No-reference · 🔁 Domain adaptation
> 🔗 Subproject: <https://openi.pcl.ac.cn/OpenPCQA/IT-PCQA>

<p align="center"><img src="IT-PCQA.png" alt="IT-PCQA architecture" width="720"/></p>
<p align="center"><em>Figure 3 · Network structure of IT-PCQA</em></p>

**Idea.** Treats natural images as the source domain and point clouds as the target domain. Quality-prediction capability is transferred from images to point clouds via unsupervised adversarial domain adaptation, removing the need for expensive subjective PCQA labels.

**Keywords:** point cloud quality assessment · no-reference · domain adaptation

#### 📂 File structure

```
root/
├── config/            # dataset configs adapted to our environment
├── MindSpore/         # code + results + models
├── TensorFlow/        # code + results + models
├── PyTorch/           # results + models; source: https://github.com/Qi-Yangsjtu/IT-PCQA
└── paper.pdf          # 2022 No-Reference Point Cloud Quality Assessment via Domain Adaptation
```

#### ⚙️ Environment

Same as VQA-PC.

**Datasets**
- [SJTU](https://smt.sjtu.edu.cn/database/point-cloud-subjective-assessment-database/)
- [TID2013](https://www.ponomarenko.info/tid2013.htm)

#### ▶️ Run

```bash
# MindSpore
cd MindSpore

# TensorFlow
cd TensorFlow

# train
python train.py
```

#### 📊 Benchmark — SJTU dataset

| Source | PLCC | SROCC | GPU memory (MB) |
|:--|:-:|:-:|:-:|
| Paper | 0.58 | 0.63 | — |
| PyTorch | 0.686 | 0.6285 | **3750** |
| MindSpore | **0.7228** | 0.5198 | 5200 |
| **TensorFlow** | 0.7221 | **0.6348** | 4750 |

> Jointly considering PLCC and SROCC, the TensorFlow version performs best overall. PyTorch uses the least GPU memory; MindSpore uses the most.

#### 📚 Citation

```bibtex
@inproceedings{yang2022ITPCQA,
  title    = {No-Reference Point Cloud Quality Assessment via Domain Adaptation},
  author   = {Yang, Qi and Liu, Yipeng and Chen, Siheng and Xu, Yiling and Sun, Jun},
  booktitle= {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year     = {2022}
}
```

**Maintainer:** Ye Hua · `yeh@pcl.ac.cn`

---

### 1.4 PQA-Net

> 📅 **2021 TCSVT** · 🏷️ Projection-based · 🎯 No-reference
> 🔗 Subproject: <https://openi.pcl.ac.cn/OpenPCQA/PQA-Net>

<p align="center"><img src="PQA-Net.png" alt="PQA-Net architecture" width="720"/></p>
<p align="center"><em>Figure 4 · Network structure of PQA-Net</em></p>

**Idea.** Deep no-reference PCQA via multi-view projection. Original PyTorch implementation: <https://github.com/qdushl/PQA-Net>.

**Keywords:** point cloud quality assessment · no-reference

#### ⚙️ Environment

- **MindSpore:** Ubuntu 16.04 · Python 3.7 · `mindspore==2.0` ([install](https://www.mindspore.cn/install))
- **TensorFlow:** Ubuntu 16.04 · Python 3.7 · `tensorflow-gpu==2.x`

**Dataset:** download `distortion.zip` from "数据集" and place it in the specified path.

#### ▶️ Run

```bash
# MindSpore
cd ./PQA-Net-mindspore
python MainDTLQ.py
python MainLQ.py

# TensorFlow
cd ./PQA-Net-tf
python distortion.py
python regression.py
```

#### 📊 Benchmark

<p align="center"><img src="PQA-Net_performance.jpg" alt="PQA-Net performance" width="640"/></p>

#### 📚 Citation

```bibtex
@article{liu2021pqa,
  title  = {PQA-Net: Deep No Reference Point Cloud Quality Assessment via Multi-view Projection},
  author = {Liu, Qi and Yuan, Hui and Su, Honglei and Liu, Hao and Wang, Yu and Yang, Huan and Hou, Junhui},
  journal= {IEEE Trans. Circuits and Systems for Video Technology},
  year   = {2021},
  doi    = {10.1109/TCSVT.2021.3100282}
}
```

**Maintainers:** Zhang Yongchi · `zhangych02@pcl.ac.cn` · Haohui Li

---

### 1.5 ResSCNN

> 📅 **2023 ACM TOMM** · 🏷️ Model-based · 🎯 No-reference · ⚡ Sparse convolution
> 🔗 Subproject: <https://openi.pcl.ac.cn/OpenPCQA/ResSCNN>

<p align="center"><img src="ResSCNN.png" alt="ResSCNN architecture" width="720"/></p>
<p align="center"><em>Figure 5 · Network structure of ResSCNN</em></p>

**Idea.** End-to-end NR PCQA using residual sparse convolutional networks on raw point clouds. Original PyTorch implementation: <https://github.com/lyp22/ResSCNN>.

**Keywords:** point cloud quality assessment · no-reference

#### ⚙️ Environment

- **MindSpore:** Ubuntu 16.04 · Python 3.7 · `mindspore==2.0` ([install](https://www.mindspore.cn/install))
- **TensorFlow:** Ubuntu 16.04 · Python 3.7 · `tensorflow-gpu==2.x`

**Dataset:** [LS-PCQA](https://sjtueducn-my.sharepoint.com/personal/liuyipeng_sjtu_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fliuyipeng%5Fsjtu%5Fedu%5Fcn%2FDocuments%2Fdistortion)

#### ▶️ Run

```bash
# MindSpore
cd ./ResSCNN-mindspore
python main.py

# TensorFlow
cd ./ResSCNN-tf
python main.py
```

#### 📊 Benchmark

<p align="center"><img src="ResSCNN_performance.jpg" alt="ResSCNN performance" width="640"/></p>

#### 📚 Citation

```bibtex
@article{Liu2022ResSCNN,
  title  = {Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Metric},
  author = {Liu, Yipeng and Yang, Qi and Xu, Yiling and Yang, Le},
  journal= {ACM Trans. Multimedia Computing, Communications and Applications},
  year   = {2022}
}
```

**Maintainers:** Zhang Yongchi · `zhangych02@pcl.ac.cn` · Haohui Li

---

## 👥 Contributors

**Coordinator:** Asst. Prof. **Wei Gao** — Shenzhen Graduate School, Peking University

| Role | Name | Affiliation |
|------|------|-------------|
| Coordinator | Asst. Prof. Wei Gao | SGS, Peking University |
| Advisor | Prof. Ge Li | SGS, Peking University |
| Contributor | Hua Ye | Peng Cheng Laboratory |
| Contributor | Yongchi Zhang | Peng Cheng Laboratory |
| Contributor | Wenxu Gao | SGS, Peking University |
| Contributor | Haohui Li | SGS, Peking University |

> Want to contribute? See [Contact](#-contact) below.

---

## 📬 Contact

If you have suggestions for improving this library, or would like to contribute your own PCQA implementation, please contact the coordinator:

**Wei Gao** — 📧 <gaowei262@pku.edu.cn>

---

<div align="center">

⭐ **If you find OpenPCQA useful, please consider starring this repository.** ⭐

</div>
