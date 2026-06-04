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
@article{liu2023point,
  title={Point cloud quality assessment: Dataset construction and learning-based no-reference metric},
  author={Liu, Yipeng and Yang, Qi and Xu, Yiling and Yang, Le},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={19},
  number={2s},
  pages={1--26},
  year={2023},
  publisher={Association for Computing Machinery}
}

## contributors

name: Zhang Yongchi && Haohui Li
email: zhangych02@pcl.ac.cn
