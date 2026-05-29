import os
import open3d as o3d
import numpy as np
import pandas as pd
import random
from scipy.linalg import expm, norm
import mindspore as ms
import mindspore.dataset as ds
import xlrd

import sys
sys.path.append("/code/ResSCNN-mindspore/")

import lib.transforms as t

from config import get_config
from lib.utils import normalize_point_coordinates, compute_voxel_grid

from itertools import repeat
from typing import List, Tuple, Union


# def ravel_hash(x: np.ndarray) -> np.ndarray:
#     assert x.ndim == 2, x.shape

#     x = x - np.min(x, axis=0)
#     x = x.astype(np.uint64, copy=False)
#     xmax = np.max(x, axis=0).astype(np.uint64) + 1

#     h = np.zeros(x.shape[0], dtype=np.uint64)
#     for k in range(x.shape[1] - 1):
#         h += x[:, k]
#         h *= xmax[k + 1]
#     h += x[:, -1]
#     return h


# def sparse_quantize(coords,
#                     voxel_size: Union[float, Tuple[float, ...]] = 1,
#                     *,
#                     return_index: bool = False,
#                     return_inverse: bool = False) -> List[np.ndarray]:
#     if isinstance(voxel_size, (float, int)):
#         voxel_size = tuple(repeat(voxel_size, 3))
#     assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

#     voxel_size = np.array(voxel_size)
#     coords = np.floor(coords / voxel_size).astype(np.int32)

#     _, indices, inverse_indices = np.unique(ravel_hash(coords),
#                                             return_index=True,
#                                             return_inverse=True)
#     coords = coords[indices]

#     outputs = [coords]
#     if return_index:
#         outputs += [indices]
#     if return_inverse:
#         outputs += [inverse_indices]
#     return outputs[0] if len(outputs) == 1 else outputs



def read_xlrd(excelFile):
  data = xlrd.open_workbook(excelFile)
  table = data.sheet_by_index(0)
  dataFile = []
  for rowNum in range(table.nrows):
    if rowNum > 0:
      dataFile.append(table.row_values(rowNum))
  return dataFile

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def lossname2dirname(loss_name):
    ret_name = loss_name
    if loss_name in ["limitlossyG-lossyA", "losslessG-limitlossyA", "losslessG-lossyA"]:
        ret_name = "AVS_" + ret_name
    elif loss_name in ["lossless-geom-lossy-attrs", "lossless-geom-nearlossless-attrs", "lossy-geom-lossy-attrs"]:
        ret_name = "GPCC_" + ret_name
    elif loss_name == "C2AI-lossy-geom-lossy-attrs":
        ret_name = "VPCC_lossy-geom-lossy-attrs"
    elif loss_name == "Octree":
        ret_name = "octree"
    
    return ret_name


class ResDataset:
    def __init__(self, phase, random_rotation=False, random_scale=False, transform=None, config=None):

        self.config = config
        self.phase = phase
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.transform = transform

        self.data_files = {
            'train': self.config.train_file,
            'test': self.config.test_file
        }[phase]

        self.files = read_xlrd(self.data_files)

        self.file_dir = self.config.file_path
        
        self.randg = np.random.RandomState()
        self.rotation_range = self.config.rotation_range

        self.min_scale = self.config.min_scale
        self.max_scale = self.config.max_scale

        self.norm_size = 1000
        self.voxel_size = self.config.voxel_size

        assert self.norm_size % self.voxel_size == 0
        
        

    def __len__(self):
        return len(self.files)
    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts
    
    def __getitem__(self, idx):
        
        pcname, moslabel = self.files[idx][0], self.files[idx][1]
        #print('pc_path:', pc_path)
        dirname = pcname.split('_')[-2]
        dirname = lossname2dirname(dirname)
        pc_file = os.path.join(self.file_dir, dirname, pcname)
        
        #print('-------------------------------------->', pc_file)

        ply = o3d.io.read_point_cloud(pc_file)

        coords = np.asarray(ply.points)
        colors = np.asarray(ply.colors)
        colors -= 0.5
        #print(colors.shape)
        
        
#         _, indices = sparse_quantize(coords=coords, voxel_size=self.voxel_size, return_index=True)
#         coords = coords[indices]
#         colors = colors[indices]
        
        coords = normalize_point_coordinates(coords, self.norm_size)
        
        if self.phase == 'train':
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
        voxel = compute_voxel_grid(coords, colors, vsize, self.voxel_size)   # (vsize, vsize, vsize, 3)
        
        voxel = np.transpose(voxel, (3, 0, 1, 2)).astype(np.float32)
        
        
        return voxel, moslabel
    
    

      
      
      

      
def get_dataloader(config, phase):
    
    transforms = []

    use_random_rotation = config.use_random_rotation and (phase in ['train'])
    use_random_scale = config.use_random_scale and (phase in ['train'])
    transforms += [t.Jitter()]

    dataset = ResDataset(
        phase,
        random_rotation=True,
        random_scale=False,
        transform=t.Compose(transforms),
        config=config
    )
    
    if phase == 'train':
        dataloader = ds.GeneratorDataset(dataset, column_names=["voxel", "moslabel"], num_parallel_workers=4, shuffle=True, python_multiprocessing=True) 
    
    else:
        dataloader = ds.GeneratorDataset(dataset, column_names=["voxel", "moslabel"], num_parallel_workers=4, shuffle=False, python_multiprocessing=True) 

    return dataloader
    

if __name__=="__main__":

    config = get_config()

    dataset = get_dataloader(config, "train")

    item = next(iter(dataset.dataset))
    print(item)
    