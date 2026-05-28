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


