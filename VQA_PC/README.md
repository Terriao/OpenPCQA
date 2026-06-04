# VQA_PC

key words:point cloud quality assessment, moving camera videos, no-reference, video quality assessment

VQA_PC is a kind of point cloud quality assessment method of no reference. It has excellent performances over SJTU, WPC and LSPCQA-I datasets, benefitting from projecting the point cloud files to videos and assessing the quality via VQA methods.

## our contributions
1.transplant from pytorch to tensorflow and mindspore.
2.benchmark test on mindspore, tensorflow and pytorch, and compare the performance.

## file structure
root
└── mindspore: mindspore code, models included
└── tensorflow: tensorflow code, models included
└── pytorch source code: please go to: https://github.com/zzc-1998/VQA_PC
└── Treating Point Cloud as Moving Camera Videos A No-Reference Quality Assessment Metric.pdf: origional paper

## datasets
1. [WPC](https://github.com/qdushl/Waterloo-Point-Cloud-Database)
2. [LS-PCQA](https://smt.sjtu.edu.cn/database/large-scale-point-cloud-quality-assessment-dataset-ls-pcqa/
)
3. [SJTU](https://smt.sjtu.edu.cn/database/point-cloud-subjective-assessment-database/)

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
> cd mindspore
* tensorflow:
> cd TensorFlow

training:
>python ./train/train_SJTU.py

test:
>python ./test/test.py

## performance
* Benchmark test on mindspore, tensorflow and pytorch below. From the result in all, we can see that the performances of versions between pytorch, mindspore and tensorflow is similar. For SJTU dataset, mindspore version has the best performance, while for WPC dataset, tensorflow version gets the best score.
* For running speed and gpu memory occupancy, mindspore takes the least time, while pytorch ranks the second with a little more time and less gpu memory. Tensorflow takes the longest time and the most volume of gpu memory.

Table 1. Test on SJTU dataset
|Source|SRCC|PLCC|KRCC|RMSE|
|:--|:--|:--|:--|:--|
|Paper|0.8509|0.8635|0.6585|1.1334|
|PyTorch|0.9125|0.9341|0.7634|0.8364|
|MindSpore|0.9136|0.9346|0.7619|0.835|
|TensorFlow|0.8774|0.9002|0.7048|1.0253|

Table 2. Test on WPC dataset
|Source|SRCC|PLCC|KRCC|RMSE|
|:--|:--|:--|:--|:--|
|Paper|0.7968|0.7976|0.6115|13.6219|
|PyTorch|0.8173|0.8226|0.631|12.9618|
|MindSpore|0.8069|0.8085|0.6188|13.4193|
|TensorFlow|0.8296|0.8313|0.6457|12.6353|

Table 3. test time and gpu memory on TESLA T4 gpu
|framework|test time(s)|GPU memory(MB)|
|:--|:--|:--|
|Paper|--|--|
|PyTorch|29.937|1358|
|MindSpore|26.405|3110|
|TensorFlow|44.887|4732|

## citation
```
@article{zhang2023evaluating,
  title={Evaluating point cloud from moving camera videos: A no-reference metric},
  author={Zhang, Zicheng and Sun, Wei and Zhu, Yucheng and Min, Xiongkuo and Wu, Wei and Chen, Ying and Zhai, Guangtao},
  journal={IEEE Transactions on Multimedia},
  volume={27},
  pages={927--939},
  year={2023},
  publisher={IEEE}
}
```
## contributors
name: Ye Hua
email: yeh@pcl.ac.cn
