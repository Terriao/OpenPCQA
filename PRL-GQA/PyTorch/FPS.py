import torch
import torch.nn.functional as F
import numpy as np
import h5py
from plyfile import PlyData
import os

'''
@author leon
@desc 采样策略
@date 2021/12
'''

def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)    # data.shape = (B,N,3), label.shape = (B,1)

def load_ply(ply_filename):     #尺寸为NX3，因此使用后最好reshape成三维
    plydata=PlyData.read(ply_filename)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    tmp = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=-1)
    tmp = np.concatenate((tmp,z.reshape(-1,1)),axis=-1)
    center = np.mean(tmp)
    tmp = tmp - center
    return tmp
def index_points(points, idx):    #将点集按照序号采样恢复成具体点坐标
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # list=[B,1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1        # list=[1,s]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)   # [B S]
    new_points = points[batch_indices, idx, :]    #batch_indices与idx维度一致，batch_indices为[[0,0,0,0,0,...][1,1,1,,,,]....[B-1,B-1......] ]   idx为[[3,8,38,6,4..][....],[....]]
    return new_points


def draw_points(points, idx):   #可视化时用
    device = points.device
    B, N, C = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    labels = torch.ones((B, N, 1)).to(device)
    labels[batch_indices, idx, :] = 0
    points_np = np.array(points.cpu())
    labels_np = np.array(labels.cpu())
    drawed_points = np.concatenate((points_np, labels_np), axis=-1)
    return drawed_points


def farthest_point_sample(xyz, npoint):  #最远点采样得到序号  BXS
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)     # xyz.shape = (B,N,3), centroid.shape = (B,1,3)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # torch.max -> value,index
    return centroids

def sample_index_save(in_path,out_path,samply_numbers):   #生成FPS采样点序号并存储
    xyz=load_ply(in_path).reshape(1,-1,3)
    xyz = torch.from_numpy(xyz).to('cuda')
    centroids = farthest_point_sample(xyz, samply_numbers)
    centroids=np.array(centroids.cpu())
    np.save(out_path,centroids)    #注意生成的采样序号的维度为B X samply_numbers,例如1X512


def FPS_sample_for_data(root_path,samply_number): #为数据集做patch中心点采样
    data = os.listdir(root_path)
    size = len(data)  # 模型总数
    for i in range(0, size):
        model_dir = os.listdir(root_path + '/' + data[i])
        for j in model_dir:
            path_j = root_path + '/' + data[i] + '/' + j
            if os.path.isfile(path_j) and j.endswith(".ply"):
                out_path=root_path + '/' + data[i] + '/'+j[0:-4]+'.npy'   #文件存储采样得到的中心点序号
                sample_index_save(path_j,out_path,samply_number)


def KNN_sample(source_set: torch.Tensor, center_set: torch.Tensor, patch_size: int):
    """
    :param source_set: size(B, N, 3)   to('cuda')
    :param center_set: size(B, S, 3)   to('cuda')
    :param patch_size: nearest k vector
    :return: size(B ,S ,patch_size)
    """
    device = source_set.device
    B = source_set.shape[0]
    S = center_set.shape[1]
    result = torch.zeros(B, S, patch_size, dtype=torch.long).to(device)
    for b in range(B):
        source2d, center2d = source_set[b], center_set[b]
        for i in range(S):
            L2_distance = torch.norm(center2d[i] - source2d, dim=-1)
            # sorted=True 即返回的下标已经按值的大小顺序排过序
            min_val, min_idx = torch.topk(L2_distance, patch_size, largest=False, sorted=True)
            result[b, i] = min_idx
    return result

def patch_sample_for_data(root_path,patch_size):   #为数据集根据中心点采样patch序号
    data = os.listdir(root_path)
    size = len(data)  # 模型总数
    for i in range(0, size):
        model_dir = os.listdir(root_path + '/' + data[i])
        center_set_list=[]
        for j in model_dir:   #某个模型的文件夹列表
            path_j = root_path + '/' + data[i] + '/' + j
            if os.path.isfile(path_j) and j.endswith(".ply"):   #原始模型文件
                source_set=load_ply(path_j).reshape(1,-1,3)   #默认只有三个坐标值
                source_set=torch.from_numpy(source_set).to('cuda')
                refer_set=source_set
                # refer_set = torch.from_numpy(refer_set).to('cuda')
                center_index=np.load(root_path + '/' + data[i] + '/'+j[0:-4]+'.npy')   #采样中心点
                center_index=torch.from_numpy(center_index).to('cuda')
                center_set=index_points(refer_set,center_index)
                center_set_list.append(center_set.cpu()) #保存一下中心点值
                result=KNN_sample(source_set,center_set,patch_size)
                out_path=root_path + '/' + data[i] + '/' +j[0:-4]+'_sample_patch.npy'
                np.save(out_path,np.array(result.cpu()))
        center_set=center_set_list[0].to('cuda')  #中心点集
        for j in model_dir:
            if os.path.isdir(root_path + '/' + data[i] + '/' + j):  # 某一噪声类型的文件夹
                for k in os.listdir(root_path + '/' + data[i] + '/' + j):
                    path = root_path + '/' + data[i] + '/' + j + '/' + k  # 某一噪声水平的模型
                    if k.endswith(".ply"):
                        source_set = load_ply(path).reshape(1,-1,3)
                        source_set = torch.from_numpy(source_set).to('cuda')
                        result=KNN_sample(source_set,center_set,patch_size)
                        out_path = root_path + '/' + data[i] + '/'+ j + '/' + k[0:-4]+'_sample_patch.npy'
                        np.save(out_path,np.array(result.cpu()))

##  固定半径内采样点
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]  为src中每一个点与dst中每一个点计算欧式距离，N行代表src的N个点，M列为与dst的M个点的距离
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))    # matmul 当有多维时，高维作为batch，低的两维作为矩阵乘法，permute转置
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    为了尽可能多的将半径内的点踩出，应尽量将点数设置大点，而半径小点，如果半径过大，点数太少，则取出的点会显的不再一个范围内。点数设置不应该超过总点数。
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]      为new_xyz中每一点向xyz中点求取距离半径内点的序号，末尾序号为0的点代表超过半径外的点.最大点数下，可能所有点都小于半径，也可能部分大于半径
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # B x S X N  [[[0,1...,N-1],[0,1,....N-1]..]]
    sqrdists = square_distance(new_xyz, xyz)   # B x S x N
    group_idx[sqrdists > radius ** 2] = N      # 距离大于半径的位置值设为N，即半径之外的点的序号设为N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]    # B x S x nsample 先按照行排序，半径之外的点肯定排在最后，且只取了最大点数值,半径范围内如果达到点数，则取最大点数，否则取半径外点以达到最大点数
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]     #  前述凑成了最大点数，此处再根据是否超出半径，将超过半径的点直接置为第一点序号，即最终序号由半径内点的序号和重复的首点序号组成，
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
       此过程不仅返回最远点采样的中心点值，还将低维特征与高维特征合并
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]   对xyz中心点采样后，    最原始的坐标点云
        points: input points data, [B, N, D]         另一高维点集，主要是上一次采点后输入到pointnet中得到的高维点集
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]    FPS中心点坐标
        new_points: sampled points data, [B, npoint, nsample, 3+D]  新采样后的最后一维为原始坐标维度加 points相同位置的点的特征维度
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]  最远点采样
    new_xyz = index_points(xyz, fps_idx)    # 序号转为坐标 B S C
    idx = query_ball_point(radius, nsample, xyz, new_xyz)    # 距离中心点半径范围内的点
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]   #   根据序号转化为坐标
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)    # 圆周采样的每个点减去采样中心点，即转化为相对中心点的相对坐标

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D] 原有维度特征拼接采样后的相对坐标
    else:
        new_points = grouped_xyz_norm   # 第一次采样，只有坐标维度
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def relative_cordinate(xyz,centroids):
    """
     将绝对坐标转化为相对于中心点的相对坐标
    :param xyz: B x S x patch_size x C
    :param centroids: B x S x C
    :return: B x S x patch_size x C
    """
    B,S,C = centroids.shape
    return xyz - centroids.view(B,S,1,C)



if __name__ == '__main__':
    # 数据预采样
    # root_path = './data5'
    # FPS_sample_for_data(root_path,72)
    # patch_sample_for_data(root_path,256)
    '''
    points = np.arange(2*5*3, dtype=np.float32).reshape((2,5,3))
    idx = np.arange(2*2, dtype=np.int64).reshape((2,2))
    points = torch.from_numpy(points)
    idx = torch.from_numpy(idx)
    out = index_points(points, idx)
    print(out)
    '''
    np.random.seed(1)
    xyz = np.random.randn(1,6,3).astype(np.float32)
    new_xyz = np.random.randn(1,4,3).astype(np.float32)
    xyz = torch.from_numpy(xyz)
    new_xyz = torch.from_numpy(new_xyz)
    out = query_ball_point(radius=3, nsample=3, xyz=xyz, new_xyz=new_xyz)
    print(out)