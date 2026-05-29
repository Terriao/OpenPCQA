import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import os

class ImageGenerator(object):
    def __init__(self, csv_file_dist, root_dir_dist, transform=None, num_plane=6):
        self.csv_file_dist = csv_file_dist
        self.root_dir_dist = root_dir_dist
        self.transform = transform
        self.num_plane = num_plane

        self.frame = pd.read_csv(self.csv_file_dist, header=None).to_numpy()


    def __getitem__(self, idx):
        filename, score, dis_type = self.frame[idx][0], self.frame[idx][1], self.frame[idx][2]
        imgs = []

        name = filename[2:-5] # Default csv
        # name = filename[:-5]
        for i in range(6):
            tname = name + str(i+1) + ".png"
            img = Image.open(os.path.join(self.root_dir_dist, tname))
            img = self.crop_image(img, (0.5, 0.5), 4)
            if self.transform:
                img = self.augment(img)
            img = np.float32(img)
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            imgs.append(img)
        images = tf.stack(imgs, 0)
        
        score = tf.convert_to_tensor(score, dtype=tf.float32)
        dis_type = tf.convert_to_tensor(dis_type, dtype=tf.int64)

        return images, score, dis_type 


    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        return self.frame.shape[0]

    def crop_image(self, img, xy, scale_factor):
        '''Crop the image around the tuple xy

        Inputs:
        -------
        img: Image opened with PIL.Image
        xy: tuple with relative (x,y) position of the center of the cropped image
            x and y shall be between 0 and 1
        scale_factor: the ratio between the original image's size and the cropped image's size
        '''
        center = (img.size[0] * xy[0], img.size[1] * xy[1])
        new_size = (img.size[0] / scale_factor, img.size[1] / scale_factor)
        left = max(0, (int)(center[0] - new_size[0] / 2))
        right = min(img.size[0], (int)(center[0] + new_size[0] / 2))
        upper = max(0, (int)(center[1] - new_size[1] / 2))
        lower = min(img.size[1], (int)(center[1] + new_size[1] / 2))
        cropped_img = img.crop((left, upper, right, lower))
        return cropped_img

    def augment(self, x):
        # x = tf.image.central_crop(x, 235/x.size[0])
        x = centercrop(x, (235, 235))
        x = random_horizontal_flip(x)
        x = random_vertical_flip(x)
        return x

def centercrop(img, size):
    new_width, new_height = size
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

def random_horizontal_flip(img, p=0.5):
    if np.random.random() < p:
        return tf.image.flip_left_right(img)
    else:
        return img


def random_vertical_flip(img, p=0.5):
    if np.random.random() < p:
        return tf.image.flip_up_down(img)
    else:
        return img


class ImageDataset(object):
    def __init__(self, csv_file_dist, root_dir_dist, batch_size, epoch, shuffle=False,
        transform=None, num_plane=6
        ):
        self.csv_file_dist = csv_file_dist
        self.root_dir_dist = root_dir_dist
        self.batch_size = batch_size
        self.transform = transform
        self.num_plane = num_plane

        self.generator = ImageGenerator(csv_file_dist=self.csv_file_dist,
                                        root_dir_dist=self.root_dir_dist,
                                        transform=self.transform,
                                        num_plane=self.num_plane
                                        )

        if shuffle:
            self.idx = np.arange(len(self.generator))
            np.random.shuffle(self.idx)
            self.generator.frame = self.generator.frame[self.idx]
        
        ot = (tf.float32, tf.float32, tf.int64)
        self.dataset = tf.data.Dataset.from_generator(self.generator, ot)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=shuffle)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.repeat(epoch)
        

    def __len__(self):
        return len(self.generator) // self.batch_size



