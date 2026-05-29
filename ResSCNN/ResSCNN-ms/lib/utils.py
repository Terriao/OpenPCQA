import open3d as o3d
import mindspore as ms
import numpy as np
import os

class Printl(object):
    def __init__(self, file) -> None:
        self.file = file
        if os.path.exists(self.file):
            os.remove(self.file)

    def __call__(self, info):
        print(info)
        if self.file:
            with open(self.file, "a") as f:
                print(info, file=f)

def normalize_point_coordinates(coors, norm_size):

    min_coor_axis0 = np.min(coors, axis=0, keepdims=True)
    coors -= min_coor_axis0
    max_coor = np.max(coors)

    assert max_coor != 0

    coors /= max_coor
    coors *= norm_size
    
    return coors

def compute_voxel_grid(coords, colors, vsize, voxel_size):
    
    voxels = np.zeros((vsize, vsize, vsize, 3), dtype=float)

    coords = np.floor(coords/voxel_size)
    coords -= (coords == vsize)
    coords = coords.astype("int")

    voxels[coords[:,0], coords[:, 1], coords[:, 2], :] = colors
    
    return voxels


if __name__=="__main__":
    pass

    # o3d_imp(f, 5)
