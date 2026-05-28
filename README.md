![](openpcqa.png)
## OpenPCQA
OpenPCQA is An Open-Source Algorithm Library of Point Cloud Quality Assessment (PCQA) based on Deep Learning. We collect methods on PCQA, provide source codes of MindSpore, PyTorch, TensorFlow, and test their performances.

## Contact and References
Coordinator: Asst. Prof. Wei Gao (Shenzhen Graduate School, Peking University)  
Should you have any suggestions for better constructing this open source library, please contact the coordinator via Email: gaowei262@pku.edu.cn. We welcome more participants to submit your codes to this collection, and you can send your OpenI ID to the above Email address to obtain the accessibility. 

## List of Contributors
Contributors:
Asst. Prof. Wei Gao (Shenzhen Graduate School, Peking University)  
Prof. Ge Li (Shenzhen Graduate School, Peking University)  
Mr. Hua Ye (Peng Cheng Laboratory)  
Mr. Yongchi Zhang (Peng Cheng Laboratory)  
Mr. Wenxu Gao (Shenzhen Graduate School, Peking University)  
Mr. Haohui Li (Shenzhen Graduate School, Peking University)  
etc.

## Table of Content
1.1 VQA-PC  
1.2 PRL-GQA  
1.3 IT-PCQA  
1.4 PQA-Net  
1.5 ResSCNN  

### 1.1 VQA-PC
- 2022 Sep 11. No-reference (NR) PCQA method, projection-based.
- Dealing with PCQA tasks via using video quality assessment (VQA) methods. Extracting both spatial and temporal quality-aware features from the selected key frames and the video clips through using trainable 2D-CNN and pretrained 3D-CNN models respectively.
- Code in the framework of tensorflow & pytorch & mindspore are provided.
- For more information, please go to [VQA-PC](https://openi.pcl.ac.cn/OpenPCQA/VQA_PC).
![VQA-PC](VQA-PC.png)
Figure 1: Network structure of VQA-PC, from Ref. [Zhang, Z., Sun, W., Min, X., Fan, Y., & Zhai, G. (2022). Treating Point Cloud as Moving Camera Videos: A No-Reference Quality Assessment Metric. arXiv preprint arXiv:2208.14085. etc.]
### 1.2 PRL-GQA
- 2022 Nov 2. No-reference (NR) PCQA method, model-based.
- The first pairwise learning framework for no-reference geometry-only quality assessment of point clouds. Takes as input a pair of point clouds and outputs their rank order.
- Code in the framework of tensorflow & pytorch & mindspore are provided.
- For more information, please go to [PRL-GQA](https://openi.pcl.ac.cn/OpenPCQA/PRL-GQA).
![PRL-GQA](PRL-GQA.png)
Figure 2: Network structure of PRL-GQA, from Ref. [Su, Z., Chu, C., Chen, L., Li, Y., & Li, W. (2022). No-reference Point Cloud Geometry Quality Assessment Based on Pairwise Rank Learning. arXiv preprint arXiv:2211.01205.]
# PRL-GQA

key words: point cloud, geometry quality assessment, rank learning, objective quality assessment, point cloud quality assessment

PRL-GQA is a no-reference geometry-only point cloud quality assessment method. It leverages the pairwise rank learning to predict relative geometry quality order and exhibits competitive ranking accuracy on the proposed PRLD dataset.

## our contributions
1.transplant from pytorch to tensorflow and mindspore.
2.benchmark test on mindspore, tensorflow and pytorch, and compare the performance.

## file structure
root
└── MindSpore: mindspore code, models included
└── TensorFlow: tensorflow code, models included
└── PyTorch: models in pytorch version, for pytorch source code, please go to: https://zhiyongsu.github.io/Project/PRLGQA.html
└── No-reference Point Cloud Geometry Quality Assessment Based on Pairwise Rank Learning.pdf: origional paper

## datasets
[PRLD]( https://zhiyongsu.github.io/Project/PRLGQA.html)

## environment
1. mindspore
- ubuntu 16.04
- cuda V11.1.105
- python 3.7.11
- mindspore 2.0.0.dev20230109, installation reference: https://www.mindspore.cn/install
2. tensorflow
- ubuntu 16.04
- cuda V10.1.243
- python 3.7.6
- tensorflow-gpu 2.3.1

## command
* mindspore:
> cd MindSpore
* tensorflow:
> cd TensorFlow

training & test:
>python ./train_test.py

## performance
* Benchmark test on MindSpore, TensorFlow and PyTorch below. From the result in all, we can see that the performances of versions between PyTorch, MindSpore and TensorFlow is similar. For PRLD dataset, PyTorch version has the best performance, while the TensorFlow version gets the worst score.
* In terms of test time and GPU memory occupancy, the PyTorch version has the fastest running speed and least GPU memory. The TensorFlow version takes a bit more GPU memory and the longest running time. The MindSpore version ranks second in running time and has the most volume of GPU memory(more than three times the GPU memory of the PyTorch version). 

Table 1. Test on PRLD dataset
|Source|Accuracy|Test Time(s)|Gpu Memory(MB)|
|:--|:--|:--|:--|
|Paper|0.9449|--|--|
|PyTorch|0.9260|624.1|2198|
|MindSpore|0.9159|1308.1|4012|
|TensorFlow|0.8924|1962.8|2692|

## citation
```
@article{su2022no,
  title={No-reference Point Cloud Geometry Quality Assessment Based on Pairwise Rank Learning},
  author={Su, Zhiyong and Chu, Chao and Chen, Long and Li, Yong and Li, Weiqing},
  journal={arXiv preprint arXiv:2211.01205},
  year={2022}
}
```

## contributors
name: Ye Hua, Gao Wenxu
email: yeh@pcl.ac.cn, gaowx@stu.pku.edu.cn

### 1.3 IT-PCQA
- 2022 CVPR. No-reference (NR) PCQA method, projection-based.
- Treating natural images as the source domain and point clouds as the target domain, and inferring point cloud quality via unsupervised adversarial domain adaptation.
- Code in the framework of tensorflow & pytorch & mindspore are provided.
- For more information, please go to [IT-PCQA](https://openi.pcl.ac.cn/OpenPCQA/IT-PCQA).
![IT-PCQA](IT-PCQA.png)
Figure 3: Network structure of IT-PCQA, from Ref. [Yang, Q., Liu, Y., Chen, S., Xu, Y., & Sun, J. (2022). No-reference point cloud quality assessment via domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 21179-21188).]

### 1.4 PQA-Net
- 2021 TCSVT. No-reference (NR) PCQA method, projection-based.
- Code in the framework of tensorflow & pytorch & mindspore are provided.
- For more information, please go to [PQA-Net](https://openi.pcl.ac.cn/OpenPCQA/PQA-Net).
![PQA-Net](PQA-Net.png)
Figure 4: Network structure of PQA-Net, from Ref. [Liu, Q., Yuan, H., Su, H., Liu, H., Wang, Y., Yang, H., & Hou, J. (2021). PQA-Net: Deep no reference point cloud quality assessment via multi-view projection. IEEE Transactions on Circuits and Systems for Video Technology, 31(12), 4645-4660.]

### 1.5 ResSCNN
- 2023 ACM T MULTIM COMPUT. No-reference (NR) PCQA method, model-based.
- Code in the framework of tensorflow & pytorch & mindspore are provided.
- For more information, please go to [ResSCNN](https://openi.pcl.ac.cn/OpenPCQA/ResSCNN).
![ResSCNN](ResSCNN.png)
Figure 5: Network structure of ResSCNN, from Ref. [Liu, Y., Yang, Q., Xu, Y., & Yang, L. (2023). Point cloud quality assessment: Dataset construction and learning-based no-reference metric. ACM Transactions on Multimedia Computing, Communications and Applications, 19(2s), 1-26.]


