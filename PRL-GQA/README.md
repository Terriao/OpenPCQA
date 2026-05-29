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