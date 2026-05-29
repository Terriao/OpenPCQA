# ResSCNN
This is the repository of ResSCNN. The original code is implemented by Pytorch, while we provide Mindspore and Tensorflow.
Key words: point cloud quality assessment, no-reference
In this case, the original implementation of PyTorch is as follows:
https://github.com/lyp22/ResSCNN

## Environment
For ResSCNN-mindspore:
ubuntu 16.04
python 3.7
mindspore 2.0, installation reference: https://www.mindspore.cn/install

For ResSCNN-tensorflow:
ubuntu 16.04
python 3.7
tensorflow-gpu 2.x

## Dataset
Link for [LS-PCQA](https://sjtueducn-my.sharepoint.com/personal/liuyipeng_sjtu_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fliuyipeng%5Fsjtu%5Fedu%5Fcn%2FDocuments%2Fdistortion) 

## Training 

For mindspore, 
```shell 
cd ./ResSCNN-mindspore
python main.py
```

For tensorflow, 
```shell 
cd ./ResSCNN-tf
python main.py
```


## Performance comparison
Benchmark test on mindspore, tensorflow and pytorch below
Paper:

![image](ResSCNN_performance.jpg)

## Citation 
@article{Liu2022ResSCNN,
title={Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Metric},
author={Yipeng Liu and Qi Yang and Yiling Xu and Le Yang},
journal={ACM Transactions on Multimedia Computing Communications and Applications},
year={2022}
}

## contributors

name: Zhang Yongchi && Haohui Li
email: zhangych02@pcl.ac.cn