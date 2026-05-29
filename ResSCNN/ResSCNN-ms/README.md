# ResSCNN

A no-reference point cloud quality assessment method using the sparse convolutional neural network.

Enviroment
-----------

python 3.7  
cudatoolkit 11.1  
pytorch 1.8  
minkowskiengine 0.5  
xlrd 1.2  
open3d  
easydict  
numpy  
scipy  
tensorboardX  
pandas  
tqdm  

Sample scale normalization
-----------
The xyz coordinates of training samples are recommended to be casted into 0-2000. If not, voxel_size in config.py should be adjusted properly.

Usage
-----------
train.py is the main program. The explanations of some parameters in config.py are as below:

trainer_arg.add_argument('--train_file', type=str, default='./config/train.xlsx', help='file name and MOS for training set')  
trainer_arg.add_argument('--test_file', type=str, default='./config/test.xlsx', help='file name and MOS for testing set')  
trainer_arg.add_argument('--train_path', type=str, default='./config/path.xlsx', help='file name and file path for training set')  
trainer_arg.add_argument('--test_path', type=str, default='./config/path.xlsx', help='file name and file path for testing set')  

To load the existing model, the related parameter in config.py is:  

misc_arg.add_argument('--resume', type=str, default='checkpoints/test_checkpoint-LS.pth', help='path for loading the checkpoint')  

ps: test_checkpoint-LS.pth is trained on LS-PCQA dataset.


Large-scale Point Cloud Quality Assessment Dataset (LS-PCQA)
-----------
We establish a large-scale point cloud quality assessment dataset named LS-PCQA includes 104 reference point clouds and more than 22,000 distorted samples. The details can be found in [our website](http://smt.sjtu.edu.cn) and our paper (Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Metric).

Link for reference point clouds: [BaiduNetDisk](https://pan.baidu.com/s/1lGB3ZGy2e6h080ItxruPYQ?pwd=4dha) [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/liuyipeng_sjtu_edu_cn/EWPixQDje2BIp_nL-6gx4qsB9LGgafD_qPpzfrzw5lz79A?e=jm5X6u)

930 distorted samples with accurate MOS are supplied.
Link: [BaiduNetDisk](https://pan.baidu.com/s/1yhyn3PZzpnuokCqZeRe0XQ?pwd=3uw3) [OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/liuyipeng_sjtu_edu_cn/Et1MMnF1JHhJiWR5Q1JoJBABlPLtBwt1oHheB4fkUK-hGw?e=e11tBG)

Link for whole dataset with generated pseudo MOS: [BaiduNetDisk](https://pan.baidu.com/s/1twG77mYdy4Knm0RpFHhqsA?pwd=garj) [OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/liuyipeng_sjtu_edu_cn/EmZV4in-Nm1Pj2wPOi2DuxQBwBMlamLVLMxRZh8aPonf2Q?e=KQjT1o)


Bibtex
-----------
If you use this code or our dataset please cite the paper

"Yipeng Liu, Qi Yang, Yiling Xu, Le Yang, "Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Metric", ACM Transactions on Multimedia
Computing Communications and Applications, 2022."

@article{Liu2022ResSCNN,  
    title={Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Metric},  
    author={Yipeng Liu and Qi Yang and Yiling Xu and Le Yang},  
    journal={ACM Transactions on Multimedia Computing Communications and Applications},  
    year={2022}  
}

