# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
# [GCC 7.5.0]
# Embedded file name: /userhome/BBBBBBBBB/PQA-Net-mindspore/ImageQualityDatasetLQSixSeperate.py
# Compiled at: 2023-01-12 15:26:02
# Size of source mod 2**32: 11319 bytes
import os
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.getcwd()

class ImageQualityDatasetLQFull:

    def __init__(self, csv_file_dist, csv_file_mos, root_dir_dist, root_dir_mos, enable_dist, transform=None, train=None, num_plane=6):
        self.enable_dist = enable_dist
        if self.enable_dist:
            self.frame = pd.read_csv(csv_file_dist, header=None)
        else:
            self.frame = pd.read_csv(csv_file_mos, header=None)
        self.root_dir_dist = root_dir_dist
        self.root_dir_mos = root_dir_mos
        self.transform = transform
        self.num_plane = num_plane
        self.csv_file_dist = csv_file_dist
        self.csv_file_mos = csv_file_mos
        self.train = train

    def __len__(self):
        return len(self.frame)

    def crop_image(self, img, xy, scale_factor):
        """Crop the image around the tuple xy

        Inputs:
        -------
        img: Image opened with PIL.Image
        xy: tuple with relative (x,y) position of the center of the cropped image
            x and y shall be between 0 and 1
        scale_factor: the ratio between the original image's size and the cropped image's size
        """
        center = (
         img.size[0] * xy[0], img.size[1] * xy[1])
        new_size = (img.size[0] / scale_factor, img.size[1] / scale_factor)
        left = max(0, int(center[0] - new_size[0] / 2))
        right = min(img.size[0], int(center[0] + new_size[0] / 2))
        upper = max(0, int(center[1] - new_size[1] / 2))
        lower = min(img.size[1], int(center[1] + new_size[1] / 2))
        cropped_img = img.crop((left, upper, right, lower))
        return cropped_img

    def __getitem__(self, idx):
        transformtensor = vision.ToTensor()
        transformtensor = vision.CenterCrop(size=235)
        imagesfull = []
        images_arr = np.array(imagesfull)
        img_set = self.frame.iloc[(idx, 0)].rsplit('_', 1)
        img_set_name = img_set[0] + '_'
        if self.enable_dist:
            csv_file_dist_full = self.csv_file_dist.rsplit('_', 1)[0]
            img_ori = csv_file_dist_full + '_dist_index.txt'
        else:
            if self.train:
                csv_file_mos_full = self.csv_file_mos.rsplit('_')
                img_ori = csv_file_mos_full + '_mos_index.txt'
            else:
                csv_file_mos_full = self.csv_file_mos.rsplit('_')
                img_ori = csv_file_mos_full[0] + '_mos_index.txt'
        ori_full_img = pd.read_csv(img_ori, header=None)
        datalist = []
        for line in range(ori_full_img.shape[0]):
            if img_set_name in ori_full_img.iloc[(line, 0)]:
                datalist.append(ori_full_img.iloc[(line, 0)][2:])
            for i in range(self.num_plane):
                if self.enable_dist:
                    img_name = os.path.join(self.root_dir_dist, datalist[i])
                else:
                    img_name = os.path.join(self.root_dir_mos, datalist[i])
                ori_image = Image.open(img_name)
                image = self.crop_image(ori_image, (0.5, 0.5), 4)
                image_arr = np.array(image)
                if i:
                    if self.transform:
                        image_temp = Image.fromarray(image_arr)
                        image_temp_img = transformcrop(image_temp)
                        image_arr = np.array(image_temp_img)
                        images_arr = np.concatenate((image_arr, images_arr), axis=2)
                    else:
                        images_arr = np.concatenate((image_arr, images_arr), axis=2)
                elif self.transform:
                    image_temp = Image.fromarray(image_arr)
                    image_temp_img = transformcrop(image_temp)
                    image_arr = np.array(image_temp_img)
                    images_arr = image_arr
                else:
                    images_arr = image_arr
            else:
                imagesfull = transformtensor(images_arr)
                score = self.frame.iloc[(idx, 1)]
                if self.enable_dist:
                    disttype = self.frame.iloc[(idx, 2)]
                    sample = {'image': 'imagesfull', 'score': 'score', 'disttype': 'disttype', 
                     'image_name': 'img_name', 'patch_num': 1}
                else:
                    sample = {
                     'image': 'imagesfull', 'score': 'score', 
                     'image_name': 'img_name', 'patch_num': 1}
                return sample


class ImageQualityDatasetLQSixSeperate:

    def __init__(self, csv_file_dist, csv_file_mos, root_dir_dist, root_dir_mos, enable_dist, transform=None, train=None, num_plane=6):
        self.enable_dist = enable_dist
        if self.enable_dist:
            self.frame = pd.read_csv(csv_file_dist, header=None)
        else:
            self.frame = pd.read_csv(csv_file_mos, header=None)
        self.root_dir_dist = root_dir_dist
        self.root_dir_mos = root_dir_mos
        self.transform = transform
        self.num_plane = num_plane
        self.csv_file_dist = csv_file_dist
        self.csv_file_mos = csv_file_mos
        self.train = train

    def __len__(self):
        return len(self.frame)

    def crop_image(self, img, xy, scale_factor):
        """Crop the image around the tuple xy

        Inputs:
        -------
        img: Image opened with PIL.Image
        xy: tuple with relative (x,y) position of the center of the cropped image
            x and y shall be between 0 and 1
        scale_factor: the ratio between the original image's size and the cropped image's size
        """
        center = (
         img.size[0] * xy[0], img.size[1] * xy[1])
        new_size = (img.size[0] / scale_factor, img.size[1] / scale_factor)
        left = max(0, int(center[0] - new_size[0] / 2))
        right = min(img.size[0], int(center[0] + new_size[0] / 2))
        upper = max(0, int(center[1] - new_size[1] / 2))
        lower = min(img.size[1], int(center[1] + new_size[1] / 2))
        cropped_img = img.crop((left, upper, right, lower))
        return cropped_img

    def __getitem__(self, idx):
        #print('idx', idx)
        #idx = (idx + 1) % 3
        transformcrop = vision.CenterCrop(size=235)
        transformhorizonflip = vision.RandomHorizontalFlip(0.5)
        transformvertiaclflip = vision.RandomVerticalFlip(0.5)
        
        imagesfull = []
        images_arr = np.array(imagesfull)

        img_set = self.frame.iloc[idx, 0].rsplit('_', 1)
        img_set_name = img_set[0] + "_"  # bag_gsigma_2_tsigma_8_view_1
        if self.enable_dist:
            csv_file_dist_full = self.csv_file_dist.rsplit('_', 1)[0]
            img_ori = csv_file_dist_full + '_mos_index.txt'
        else:
            if self.train:
                csv_file_mos_full = self.csv_file_mos.rsplit(('_', 1)[0])
                img_ori = csv_file_mos_full + '_mos_index.txt'
            else:
                csv_file_mos_full = self.csv_file_mos.rsplit(('_', 1)[0])
                img_ori = csv_file_mos_full[0] + '_mos_index.txt'
        ori_full_img = pd.read_csv(img_ori, header=None)
        # print(self.csv_file_dist)
        datalist = []
        for line in range(ori_full_img.shape[0]):
            # find all the image (which is in the xx_index.txt) projected from the same point cloud.
            if img_set_name in ori_full_img.iloc[line, 0]:
                # store all the project pictures from the same point cloud
                datalist.append(ori_full_img.iloc[line, 0][2:])
        #print('datalist:', len(datalist))
        image_arr_temp = np.zeros([self.num_plane, 235, 235, 3], dtype=int)
        images_arr_temp = np.zeros([self.num_plane, 3, 235, 235], dtype=int)
        for i in range(self.num_plane):
            if self.enable_dist:
                img_name = os.path.join(self.root_dir_dist, datalist[i])
            else:
                img_name = os.path.join(self.root_dir_mos, datalist[i])
            ori_image = Image.open(img_name)
            image = self.crop_image(ori_image, (0.5, 0.5), 4)
            image_arr = np.array(image)
            if self.transform:
                image_temp = Image.fromarray(image_arr)
                image_temp_img = transformcrop(image_temp)
                image_temp_img = transformhorizonflip(image_temp_img)
                image_temp_img = transformvertiaclflip(image_temp_img)
                image_arr_temp[i, :, :, :] = np.array(image_temp_img)
                images_arr_temp[i, :, :, :] = image_arr_temp[i, :, :, :].transpose(2, 0, 1)
            else:
                image_arr_temp[i, :, :, :] = image_arr
                images_arr_temp[i, :, :, :] = image_arr_temp[i, :, :, :].transpose(2, 0, 1)
        images_arr = np.stack([images_arr_temp[0, :, :, :], images_arr_temp[1, :, :, :], images_arr_temp[2, :, :, :], images_arr_temp[3, :, :, :], images_arr_temp[4, :, :, :], images_arr_temp[5, :, :, :]], axis=0)
        imagesfull = images_arr
        score = self.frame.iloc[(idx, 1)].astype(np.float32)
        if self.enable_dist:
            disttype = self.frame.iloc[(idx, 2)].astype(np.int32)
            sample = (imagesfull, score, disttype, img_name, 1)
        else:
            sample = (imagesfull, score, img_name, 1)
        return sample

                