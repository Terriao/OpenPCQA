import os
import numpy as np
import pandas as pd
from PIL import Image
# from torchvision import transforms
# import torch
# from torch.utils import data
# import mindspore as mindspore
# import mindspore.ops as ops
# import mindspore.dataset as ds
# from mindspore import Tensor

class VideoDataset_NR_image_with_fast_features():
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D , datainfo_path, transform, crop_size, frame_index=5, image_length_read = 4):
        # super(VideoDataset_NR_image_with_fast_features, self).__init__()
                                        
        # column_names = ['vid_name', 'scene', 'dis_type_level']
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.moss = dataInfo['mos']

        self.crop_size = crop_size
        self.data_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.frame_index = frame_index
        self.image_length_read = image_length_read

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx] 
        frames_dir = os.path.join(self.data_dir, video_name)

        mos = self.moss.iloc[idx]

        image_channel = 3
        image_height_crop = self.crop_size
        image_width_crop = self.crop_size
       
        image_length_read = self.image_length_read       
        
        # transformed_image = torch.zeros([image_length_read, image_channel, image_height_crop, image_width_crop])
        # transformed_image = ops.Zeros()((image_length_read, image_channel, image_height_crop, image_width_crop), mindspore.float32)
        transformed_image = np.zeros([image_length_read, image_channel, image_height_crop, image_width_crop], dtype=np.float32)

        image_read_index = 0
        for i in range(image_length_read):
            # select the j-th frame every 30 frames
            imge_name = os.path.join(frames_dir, str(self.frame_index+i*30).zfill(3) + '.png')
            if os.path.exists(imge_name):
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_image[i] = read_frame[0]

                image_read_index += 1
            else:
                print(imge_name)
                print('Image do not exist!')

        if image_read_index < image_length_read:
            for j in range(image_read_index, image_length_read):
                transformed_image[j] = transformed_image[image_read_index-1] #用最后一个填充

        # read 3D features
        feature_folder_name = os.path.join(self.data_dir_3D, video_name.split('.')[0])
        # transformed_feature = torch.zeros([image_length_read, 256])
        # transformed_feature = ops.Zeros()((image_length_read, 256), mindspore.float32)
        transformed_feature = np.zeros([image_length_read, 256], dtype=np.float32)
        for i in range(image_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i) + '_fast_feature.npy'))
            # feature_3D = torch.from_numpy(feature_3D)
            # feature_3D = Tensor.from_numpy(feature_3D)
            feature_3D = np.squeeze(feature_3D)
            transformed_feature[i] = feature_3D


        return transformed_image, transformed_feature, mos, video_name