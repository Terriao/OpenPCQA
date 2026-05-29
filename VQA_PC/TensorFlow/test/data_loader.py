import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

class VideoDataset_NR_image_with_fast_features():
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D , datainfo_path, crop_size, epoch, batch_size, 
        is_train=True, frame_index=5, image_length_read = 4):
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.moss = dataInfo['mos']

        self.crop_size = crop_size
        self.data_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.is_train = is_train
        self.length = len(self.video_names)
        self.frame_index = frame_index
        self.image_length_read = image_length_read

        self.dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.length))
        if is_train:
            self.dataset = self.dataset.shuffle(buffer_size=self.length, seed=epoch+1, reshuffle_each_iteration=True) #每个epoch的seed必须不一样，不然等于没shuffle
        self.dataset = self.dataset.map(lambda x: tf.numpy_function(self.parse_item, [x], [tf.float32, tf.float32, tf.float32, tf.string]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        self.dataset = self.dataset.repeat(1) #每个epoch重新创建dataset
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) #要放在batch之后

    def parse_item(self, idx):
        video_name = self.video_names.iloc[idx] 
        frames_dir = os.path.join(self.data_dir, video_name)

        mos = np.float32(self.moss.iloc[idx])

        # video_channel = 3
        # video_height_crop = self.crop_size
        # video_width_crop = self.crop_size
       
        image_length_read = self.image_length_read       
        # transformed_image = np.zeros([image_length_read, video_height_crop, video_width_crop, video_channel], dtype=np.float32)#channel放最后是为了tf.image数据处理的需要，但TensorFlow的tensor不支持直接赋值
        image_read_index = 0
        for i in range(image_length_read):
            # select the j-th frame every 30 frames 
            imge_name = os.path.join(frames_dir, str(self.frame_index+i*30).zfill(3) + '.png')
            if os.path.exists(imge_name):
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                # read_frame = self.transform(read_frame)
                read_frame = np.array(read_frame, dtype=np.float32) #(1061, 1920, 3)hwc
                read_frame = tf.convert_to_tensor(read_frame)
                h,w,c = read_frame.shape
                if self.is_train:
                    read_frame = tf.image.random_crop(read_frame, (224, 224, 3))
                else:
                    read_frame = tf.image.crop_to_bounding_box(read_frame, h//2-(224//2), w//2-(224//2), 224, 224) #CenterCrop(224)
                read_frame = read_frame / 255.0
                read_frame = (read_frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

                # transformed_image[i] = read_frame[0]
                read_frame = tf.expand_dims(read_frame, 0)
                if(i == 0):
                    transformed_image = read_frame
                else:
                    transformed_image = tf.concat([transformed_image, read_frame], 0) #[n, 224, 224, 3]

                image_read_index += 1
            else:
                print(imge_name)
                print('Image do not exist!')

        if image_read_index < image_length_read:
            for j in range(image_read_index, image_length_read):
                # transformed_image[j] = transformed_image[image_read_index-1]
                transformed_image = tf.concat([transformed_image, transformed_image[image_read_index-1:image_read_index]], 0)

        # read 3D features
        feature_folder_name = os.path.join(self.data_dir_3D, video_name.split('.')[0])
        # transformed_feature = np.zeros([image_length_read, 256], dtype=np.float32)
        for i in range(image_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i) + '_fast_feature.npy'))#(1, 256, 1, 1, 1)
            feature_3D = np.squeeze(feature_3D)#(256,)
            feature_3D = tf.convert_to_tensor(feature_3D, dtype=tf.float32)
            feature_3D = tf.expand_dims(feature_3D, 0)
            # transformed_feature[i] = feature_3D
            if(i == 0):
                transformed_feature = feature_3D
            else:
                transformed_feature = tf.concat([transformed_feature, feature_3D], 0) #[n, 256]

        return transformed_image, transformed_feature, mos, video_name
