# IT-PCQA

key words: point cloud quality assessment, no-reference, domain adaptation

IT-PCQA is a novel point cloud quality assessment method of no reference. Leveraging the rich subjective scores of the natural images, IT-PCQA quests the evaluation criteria of human perception via DNN and transfers the capability of prediction to 3D point clouds. The method suggests the feasibility of assessing the quality of specific media content without the expensive and cumbersome subjective evaluations.

## our contributions
1.transplant from pytorch to tensorflow and mindspore.
2.benchmark test on mindspore, tensorflow and pytorch, and compare the performance.

## file structure
root
└── config: dataset config files, updated from original version to fit our environment
└── MindSpore: mindspore code, results and models are included
└── TensorFlow: tensorflow code, results and models are included
└── PyTorch: results and models in pytorch version, for pytorch source code, please go to: https://github.com/Qi-Yangsjtu/IT-PCQA
└── 2022_No-Reference Point Cloud Quality Assessment via Domain Adaptation.pdf: origional paper

## datasets
1. [SJTU](https://smt.sjtu.edu.cn/database/point-cloud-subjective-assessment-database/)
2. [TID2013](https://www.ponomarenko.info/tid2013.htm)

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

training:
>python train.py

## performance
* Benchmark test on mindspore, tensorflow and pytorch below. From the result in all, we can see that the performances of versions between pytorch, mindspore and tensorflow is similar. Jointly considering the indicators of PLCC and SROCC, TensorFlow version has the best performance.
* For gpu memory occupancy of training, Mindspore takes the most volume of gpu memory and Pytorch takes the least.

Table 1. Test on SJTU dataset
|Source|PLCC|SROCC|GPU memory(MB)|
|:--|:--|:--|:--|
|Paper|0.58|0.63|-|
|Pytorch|0.686|0.6285|3750|
|Mindspore|0.7228|0.5198|5200|
|TensorFlow|0.7221|0.6348|4750|

## citation
```
"Qi Yang, Yipeng Liu, Siheng Chen, Yiling Xu, Jun Sun, "No-Reference Point Cloud Quality Assessment via Domain Adaptation," in CVPR, 2022."  
@InProceedings{yang2022ITPCQA,  
author = {Qi Yang and Yipeng Liu and Siheng Chen and Yiling Xu and Jun Sun},  
title = {No-Reference Point Cloud Quality Assessment via Domain Adaptation},  
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
year = {2022}
}
```
## contributors
name: Ye Hua
email: yeh@pcl.ac.cn
