import torch
from torch.utils.data import Dataset, DataLoader
from FPS import load_ply,farthest_point_sample,index_points,query_ball_point,relative_cordinate,draw_points,KNN_sample
import numpy as np


'''
@author leon
@desc 根据训练文档编写dataloader
注意，本文件中函数涉及的B仅仅是为了统一计算而加的，在dataloader中的get_item返回时已经去除
@date 2021/12
'''

def read_root_txt(root_txt_path):
    assert root_txt_path.endswith('.txt')     #判断文件某个条件是否成立，false引发异常
    path1_list, path2_list, labels_list ,distortion_type= [], [], [],[]
    try:
        with open(root_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
            for line in lines:
                _, path1, path2, label,distortion = line.strip().split(' ')
                label = int(label)
                path1_list.append(path1)
                path2_list.append(path2)
                labels_list.append(label)
                distortion_type.append(distortion)
    except UnicodeDecodeError:
        with open(root_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
            for line in lines:
                _, path1, path2, label,distortion = line.strip().split(' ')
                label = int(label)
                path1_list.append(path1)
                path2_list.append(path2)
                labels_list.append(label)
                distortion_type.append(distortion)
    return path1_list, path2_list, labels_list,distortion_type

# txt_full_path: 样本数据txt完整路径，读取某个路径文件
def read_sample_txt_files(txt_full_path):
    assert txt_full_path.endswith('.txt')
    with open(txt_full_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
        info = lines[0].strip()
        info = info[1: -1]
        split_infos = info.split(',')
        #print(split_infos)
        split_infos = [float(elem) for elem in split_infos]
        #split_infos =split_infos[0:45]
        return split_infos


def pc_normalize(pc):
    '''
    :param pc:B X N x D
    :return:  零均值化 B x N x D
    '''
    centroid = np.mean(pc, axis=1)  # B X D
    B = pc.shape[0]
    data =np.zeros(pc.shape,dtype=pc.dype.type)
    for batch in range(B):
        data[batch]=pc[batch]-centroid[batch]      # B x N x D
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=-1)),axis=-1)  # (B,)
    for batch in range(B):
        pc[batch] =data[batch]/m[batch]
     # 归一化到【-1 1】之间
    return pc
# def pc_normalize(pc):
#     pc :  N X C
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / m
#     return pc


def index_to_points(points, idx):
    """
    在N个点中按照序号S挑选S个值 ,与FPS中index_points功能基本一致
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
        如果是B N C与 S  可以直接写成 points[:,index,:]  idex=[0 1 2...]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    S = idx.shape[1]
    C = points.shape[2]
    new_point=np.zeros((B,S,C),dtype=points.dtype.type)
    for batch in range(B):
        points_batch=points[batch]
        new_point[batch]=points_batch[idx[batch]]
    return new_point


def random_select(model1,patch_npy1,model2,patch_npy2,random_size):
    '''
     成对模型随机挑选patch并转换为坐标值
    :param model: 点云坐标 BxNxc
    :param patch_npy: 各个patch中相对model的坐标序号 BxS x patch_size
    :param random_size: 需要挑选的patch数目
    :return: 返回挑选后的patch点  B x random_size x patch_size x c
    '''
    S=patch_npy1.shape[1]
    B=patch_npy1.shape[0]
    patch_size=patch_npy1.shape[2]
    C=model1.shape[2]
    index=np.arange(0,S)
    select_index=np.random.choice(index,random_size,replace=False)  #从总的patch数S中随机挑选一定数量进行训练输入
    select_index=select_index.reshape(-1,random_size)
    select_index=np.repeat(select_index,B,axis=0)   # [B random_size] 此操作每个Batch挑选的序号都相同
    select_patch_index1=index_to_points(patch_npy1,select_index)  # [B random_size patch_size]
    select_patch_index2=index_to_points(patch_npy2,select_index)
    result1 = np.zeros((B,random_size,patch_size,C),dtype=np.float)
    result2 = np.zeros((B, random_size, patch_size, C), dtype=np.float)
    for patch in range(random_size):
        patch_index=select_patch_index1[:,patch,:]   # [B patch_size]
        value =index_to_points(model1,patch_index)        # [B patch_size C]
        for batch in range(B):
            result1[batch][patch]=value[batch]
    for patch in range(random_size):
        patch_index=select_patch_index2[:,patch,:]   # [B patch_size]
        value =index_to_points(model2,patch_index)        # [B patch_size C]
        for batch in range(B):
            result2[batch][patch]=value[batch]
    return result1,result2


# root_txt_path: 放置路径和标签信息的txt文件完整路径
class PointCloudDataset(Dataset):
    def __init__(self, root_txt_path,catergory=False,select_patch_numbers=64):
        super(PointCloudDataset, self).__init__()
        assert isinstance(root_txt_path, str) and root_txt_path.endswith('.txt')
        # self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.sample_path1_list, self.sample_path2_list, self.labels_list,self.distortion_type = read_root_txt(root_txt_path)
        self.category=catergory   #用于是否返回噪声或者采样类型
        self.select_patch_numbers=select_patch_numbers

    def __getitem__(self, index):
        '''

        :param index: 样本序号
        :return:  random_sizexpatch_sizex3
        '''
        path1 = self.sample_path1_list[index]
        path2 = self.sample_path2_list[index]  # 样本路径
        label = self.labels_list[index]
        model1 = load_ply(path1).reshape(1,-1,3)  # ply模型    BxNx3  B=1
        model2 = load_ply(path2).reshape(1,-1,3)
        model1 = torch.from_numpy(model1)
        model2 = torch.from_numpy(model2)

        centroids_index = farthest_point_sample(model1, self.select_patch_numbers)  # 每次采样点数
        centroids = index_points(model1, centroids_index)   #确定采样中兴点坐标
        # radius采样
        result1 = query_ball_point(0.2, 516, model1, centroids)
        result2 = query_ball_point(0.2, 516, model2, centroids)   #  B x S x nsample
        result1_np = result1.numpy()
        result2_np = result2.numpy()
        B, S, patch_size = result1_np.shape
        result1_value = np.zeros((B, S, patch_size, 3), dtype=np.float)
        result2_value = np.zeros((B, S, patch_size, 3), dtype=np.float)
        model1_numpy = model1.numpy()  # 此部分代码基于numpy运算，故转换
        for patch in range(S):
            patch_index = result1_np[:, patch, :]  # [B patch_size]
            value = index_to_points(model1_numpy, patch_index)  # [B patch_size C]
            for batch in range(B):
                result1_value[batch][patch] = value[batch]  # B X S X patch_size X C
        model2_numpy = model2.numpy()  # 此部分代码基于numpy运算，故转换
        for patch in range(S):
            patch_index = result2_np[:, patch, :]  # [B patch_size]
            value = index_to_points(model2_numpy, patch_index)  # [B patch_size C]
            for batch in range(B):
                result2_value[batch][patch] = value[batch]  # B X S X patch_size X C
        data1_tensor = torch.tensor(result1_value, dtype=torch.float)
        data2_tensor = torch.tensor(result2_value, dtype=torch.float)

        # 相对坐标转换
        data1_tensor = relative_cordinate(data1_tensor, centroids)
        data2_tensor = relative_cordinate(data2_tensor, centroids)
        data1_tensor =data1_tensor[0]
        data2_tensor =data2_tensor[0]    # S X patch_size X C
        # data_patch1=pc_normalize(data_patch1)
        # data_patch2=pc_normalize(data_patch2)  #   坐标零均值化
        label_tensor = torch.tensor(label, dtype=torch.float)
        label_tensor = label_tensor.unsqueeze(-1)
        if self.category:
            return data1_tensor, data2_tensor, label_tensor,self.distortion_type[index]
        else:
            return data1_tensor, data2_tensor, label_tensor

    def __len__(self):
        return len(self.sample_path1_list)






if __name__ == '__main__':

    PCDataset = PointCloudDataset('./rank_pair_train.txt',True)
    trainloader = DataLoader(PCDataset, batch_size=1, num_workers=1, shuffle=True,
                             drop_last=False)
    for i, (sample1_tensor, sample2_tensor, label_tensor,type_tensor) in enumerate(trainloader):
        print(label_tensor)
        print(type_tensor[0]=='com&down')




   # 训练时从总固定数patch中随机挑选一定数量patch可视化结果代码
   #  root='./rank_pair_train_2.txt'
   #  sample_path1_list,sample_path2_list,labels_list = read_root_txt(root)
   #  path1=sample_path1_list[0]
   #  path2=sample_path2_list[0]
   #  model1 = load_ply(path1).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
   #  model2 = load_ply(path2).reshape(1, -1, 3)
   #  patch_npy1 = np.load(path1[0:-4] + '_sample_patch.npy')  # 采样文件BxSxpathsize
   #  patch_npy2 = np.load(path2[0:-4] + '_sample_patch.npy')
   #  S = patch_npy1.shape[1]
   #  B = patch_npy1.shape[0]
   #  patch_size = patch_npy1.shape[2]
   #  C = model1.shape[2]
   #  random_size=36
   #  index = np.arange(0, S)
   #  select_index = np.random.choice(index, random_size, replace=False)  # 从总的patch数S中随机挑选一定数量进行训练输入
   #  select_index = select_index.reshape(-1, random_size)
   #  select_index = np.repeat(select_index, B, axis=0)  # [B random_size] 此操作每个Batch挑选的序号都相同
   #  select_patch_index1 = index_to_points(patch_npy1, select_index)  # [B random_size patch_size]
   #  select_patch_index2 = index_to_points(patch_npy2, select_index)
   #  select_patch_index1=select_patch_index1.reshape(B,-1)
   #  select_patch_index2=select_patch_index2.reshape(B,-1)
   #  model1=torch.from_numpy(model1).to('cuda')
   #  model2=torch.from_numpy(model2).to('cuda')
   #  drawed_points_np1 = draw_points(model1, select_patch_index1)
   #  drawed_points_np2 = draw_points(model2, select_patch_index2)
   #  b=0
   #  np.savetxt(f'./TXT/result_with_label1{b}.txt', drawed_points_np1[b, :, :], fmt="%.5f,%.5f,%.5f,%.5f", delimiter="\n")
   #  np.savetxt(f'./TXT/result_with_label2{b}.txt', drawed_points_np2[b, :, :], fmt="%.5f,%.5f,%.5f,%.5f",delimiter="\n")



    # KNN采样示例代码
    # path1 = './G-PCD/stimuli/bunny.ply'
    # path2 = './G-PCD/stimuli/bunny_D01_L04.ply'
    # model1 = load_ply(path1).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
    # model2 = load_ply(path2).reshape(1, -1, 3)
    # model1=torch.from_numpy(model1).to('cuda')
    # model2=torch.from_numpy(model2).to('cuda')
    # centroids_index = farthest_point_sample(model1, 2)  # 每次采样点数
    # centroids = index_points(model1, centroids_index)  # 确定采样中兴点坐标
    # result1 = KNN_sample(model1, centroids, 512)  # 测试patch大小设置为256   B x S x patch_size
    # result2 = KNN_sample(model2, centroids, 512)
    # result1=result1.reshape(1,-1)
    # result2=result2.reshape(1,-1)
    # drawed_points_np1 = draw_points(model1, result1)
    # drawed_points_np2 = draw_points(model2, result2)
    # b = 0
    # np.savetxt(f'./TXT/result_with_label1{b}.txt', drawed_points_np1[b, :, :], fmt="%.5f,%.5f,%.5f,%.5f",
    #            delimiter="\n")
    # np.savetxt(f'./TXT/result_with_label2{b}.txt', drawed_points_np2[b, :, :], fmt="%.5f,%.5f,%.5f,%.5f",
    #            delimiter="\n")



    # radius采样示例
    # path1 = './data/raw_model_0/raw_model_0.ply'
    # path2 = './G-PCD/stimuli/bunny_D01_L04.ply'
    # model1 = load_ply(path1).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
    # model2 = load_ply(path2).reshape(1, -1, 3)
    # model1 = torch.from_numpy(model1).to('cuda')
    # model2 = torch.from_numpy(model2).to('cuda')
    # centroids_index = farthest_point_sample(model1, 2)  # 每次采样点数
    # centroids = index_points(model1, centroids_index)  # 确定采样中兴点坐标
    # result1 = query_ball_point(0.2, 2048, model1, centroids)
    # result2 = query_ball_point(0.2, 2048, model2, centroids)
    # result1 = result1.reshape(1, -1)
    # result2 = result2.reshape(1, -1)
    # drawed_points_np1 = draw_points(model1, result1)
    # drawed_points_np2 = draw_points(model2, result2)
    # b = 0
    # np.savetxt(f'./TXT/result_with_label1{b}.txt', drawed_points_np1[b, :, :], fmt="%.5f,%.5f,%.5f,%.5f",
    #            delimiter="\n")
    # np.savetxt(f'./TXT/result_with_label2{b}.txt', drawed_points_np2[b, :, :], fmt="%.5f,%.5f,%.5f,%.5f",
    #            delimiter="\n")


