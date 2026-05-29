import os
import open3d as o3d
import numpy as np
import pandas as pd
import random
from scipy.linalg import expm, norm
import tensorflow as tf

import sys
sys.path.append("/home/lhh/codes/ResSCNN-tf")

import lib.transforms as t

from config import get_config
from lib.utils import normalize_point_coordinates, compute_voxel_grid


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


class DataGenerator(object):
    def __init__(self, phase, transform=None, random_rotation=False, random_scale=False, config=None):
        self.config = config
        self.phase = phase
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.transform = transform

        self.data_files = {
            'train': self.config.train_file,
            'test': self.config.test_file
        }[phase]

        self.files = pd.read_csv(self.data_files).to_numpy()

        self.file_dir = self.config.file_dir

        self.randg = np.random.RandomState()
        self.rotation_range = self.config.rotation_range

        self.min_scale = self.config.min_scale
        self.max_scale = self.config.max_scale

        self.norm_size = self.config.norm_size
        self.voxel_size = self.config.voxel_size

        assert self.norm_size % self.voxel_size == 0

    def __getitem__(self, idx):

        pcname, moslabel = self.files[idx]
        moslabel = float(moslabel)

        pc_file = os.path.join(self.file_dir, pcname)

        ply = o3d.io.read_point_cloud(pc_file)

        coords = np.asarray(ply.points)
        colors = np.asarray(ply.colors)
        colors -= 0.5

        coords = normalize_point_coordinates(coords, self.norm_size)


        if self.transform:
            coords, colors = self.transform(coords, colors)

        if self.random_rotation:
            T0 = sample_random_trans(coords, self.randg, self.rotation_range)
            coords = self.apply_transform(coords, T0)
            coords = normalize_point_coordinates(coords, self.norm_size)


        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            coords = scale * coords

        vsize = self.norm_size // self.voxel_size
        voxel = compute_voxel_grid(coords, colors, vsize, self.voxel_size)

        voxel = tf.convert_to_tensor(voxel, dtype=tf.float32)
        moslabel = tf.convert_to_tensor(moslabel, dtype=tf.float32)

        return voxel, moslabel

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    
    def __len__(self):
        return self.files.shape[0]


class ResDataset(object):
    def __init__(self, phase, batch_size, epoch, random_rotation=False, random_scale=False, transform=None, shuffle=False, config=None):
        self.config = config
        self.batch_size = batch_size

        self.generator = DataGenerator(
            phase, 
            random_rotation=random_rotation,
            random_scale=random_scale,
            transform=transform, 
            config=config
        )

        if shuffle:
            self.idx = np.arange(len(self.generator))
            np.random.shuffle(self.idx)
            self.generator.files = self.generator.files[self.idx]

        ot = (tf.float32, tf.float32)
        self.dataset = tf.data.Dataset.from_generator(self.generator, ot)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=shuffle)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.repeat(epoch)

    def __len__(self):
        return len(self.generator) // self.batch_size

def get_dataset(config, phase, batch_size, epoch, shuffle=False):
    if phase == 'test':
        shuffle = 0
    
    transforms = []

    use_random_rotation = config.use_random_rotation and (phase in ['train'])
    use_random_scale = config.use_random_scale and (phase in ['train'])
    transforms += [t.Jitter()]

    dataset = ResDataset(
        phase,
        batch_size=batch_size,
        epoch=epoch,
        shuffle=shuffle,
        random_rotation=use_random_rotation,
        random_scale=use_random_scale,
        transform=t.Compose(transforms),
        config=config
    )

    return dataset
    

if __name__=="__main__":

    config = get_config()

    dataset = get_dataset(config, "train", config.batch_size, epoch=1)

    item = next(iter(dataset.dataset))
    print(item)
    