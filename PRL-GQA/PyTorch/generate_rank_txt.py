import os
import torch
'''
@author leon
@desc 用于针对data/data5文件组织形式的数据集产生pairs训练与测试文档
@date 2021/12
'''


# # data_txt_generate 生成成对训练数据集
# def data_txt_generate(_input_dir_path, _out_train_txt, _out_test_txt, _ratio_for_train):
#     data = os.listdir(_input_dir_path)
#     size = len(data)   # 模型总数
#     pair_txt_train = open(_out_train_txt, mode="w", encoding="utf-8")
#     pair_txt_test = open(_out_test_txt, mode="w", encoding="utf-8")
#     # 生成训练文档
#     for i in range(0, int(size * _ratio_for_train)):
#         model_dir = os.listdir(_input_dir_path + '/' + data[i])   # 某一模型文件夹
#         original_model = ''
#         original_model_patch=''
#         for j in model_dir:
#             path_j = _input_dir_path + '/' + data[i] + '/' + j
#             if os.path.isfile(path_j) and j.endswith(".ply"):
#                 original_model = path_j
#                 original_model_patch=_input_dir_path + '/' + data[i] + '/' + j[0:-4]+'_sample_patch.npy'
#         for j in model_dir:
#             if os.path.isdir(_input_dir_path + '/' + data[i] + '/' + j):  # 某一噪声类型的文件夹
#                 file = []
#                 for k in os.listdir(_input_dir_path + '/' + data[i] + '/' + j):
#                     path = _input_dir_path + '/' + data[i] + '/' + j + '/' + k   # 某一噪声水平的模型
#                     if k.endswith(".ply"):
#                         file.append(k)
#                         pair_txt_train.write('%s %s %s s% %s %s %d\n' %(original_model, original_model_patch, original_model, original_model_patch, path, path[0:-4]+'.npy', 1))
#                        # pair_txt_train.write('%s %s %s %d\n' % (original_model, path, original_model, 0))
#                 for k in range(len(file)):
#                     for h in range(k+1, len(file)):
#                         if file[k] > file[h]:
#                             pair_txt_train.write('%s %s s% s% %s %s %d\n' % (original_model,original_model_patch,
#                                                                     _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
#                                                                     _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k][0:-4]+'.npy',
#                                                                     _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                                                                     _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h][0:-4]+'.npy', 0))
#                             # pair_txt_train.write('%s %s %s %d\n' % (original_model,
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k], 1))
#                         else:
#                             pair_txt_train.write('%s %s s% s% %s %s %d\n' % (original_model, original_model_patch,
#                                                                              _input_dir_path + '/' + data[ i] + '/' + j + '/' + file[k],
#                                                                              _input_dir_path + '/' + data[ i] + '/' + j + '/' + file[k][ 0:-4] + '.npy',
#                                                                              _input_dir_path + '/' + data[ i] + '/' + j + '/' + file[h],
#                                                                              _input_dir_path + '/' + data[ i] + '/' + j + '/' + file[h][ 0:-4] + '.npy',
#                                                                              1))
#                             # pair_txt_train.write('%s %s %s %d\n' % (original_model,
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k], 0))
#     # 生成测试文档
#     for i in range(int(size * _ratio_for_train), size):
#         model_dir = os.listdir(_input_dir_path + '/' + data[i])  # 某一模型文件夹
#         original_model = ''
#         original_model_patch = ''
#         for j in model_dir:
#             path_j = _input_dir_path + '/' + data[i] + '/' + j
#             if os.path.isfile(path_j) and j.endswith(".ply"):
#                 original_model = path_j
#                 original_model_patch = _input_dir_path + '/' + data[i] + '/' + j[0:-4] + '_sample_patch.npy'
#         for j in model_dir:
#             if os.path.isdir(_input_dir_path + '/' + data[i] + '/' + j):  # 某一噪声类型的文件夹
#                 file = []
#                 for k in os.listdir(_input_dir_path + '/' + data[i] + '/' + j):
#                     path = _input_dir_path + '/' + data[i] + '/' + j + '/' + k  # 某一噪声水平的模型
#                     if k.endswith(".ply"):
#                         file.append(k)
#                         pair_txt_test.write('%s %s %s s% %s %s %d\n' % (
#                         original_model, original_model_patch, original_model, original_model_patch, path,
#                         path[0:-4] + '.npy', 1))
#                     # pair_txt_train.write('%s %s %s %d\n' % (original_model, path, original_model, 0))
#                 for k in range(len(file)):
#                     for h in range(k + 1, len(file)):
#                         if file[k] > file[h]:
#                             pair_txt_test.write('%s %s s% s% %s %s %d\n' % (original_model, original_model_patch,
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k][0:-4] + '.npy',
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h][0:-4] + '.npy',0))
#                             # pair_txt_train.write('%s %s %s %d\n' % (original_model,
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k], 1))
#                         else:
#                             pair_txt_test.write('%s %s s% s% %s %s %d\n' % (original_model, original_model_patch,
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k][0:-4] + '.npy',
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                                                                              _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h][0:-4] + '.npy',1))
#                             # pair_txt_train.write('%s %s %s %d\n' % (original_model,
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h],
#                             #                                         _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k], 0))



# data_txt_generate 生成成对训练数据集
def data_txt_generate(_input_dir_path, _out_train_txt, _out_test_txt, _ratio_for_train):
    data = os.listdir(_input_dir_path)
    size = len(data)   # 模型总数
    pair_txt_train = open(_out_train_txt, mode="w", encoding="utf-8")
    pair_txt_test = open(_out_test_txt, mode="w", encoding="utf-8")
    # 生成训练文档
    for i in range(0, int(size * _ratio_for_train)):
        model_dir = os.listdir(_input_dir_path + '/' + data[i])   # 某一模型文件夹
        original_model = ''
        for j in model_dir:
            path_j = _input_dir_path + '/' + data[i] + '/' + j
            # print(path_j)
            if os.path.isfile(path_j) and j.endswith(".ply"):
                original_model = path_j
                # print(original_model)
        for j in model_dir:
            if os.path.isdir(_input_dir_path + '/' + data[i] + '/' + j):  # 某一失真类型的文件夹
                distortion_type = j
                file = []
                for k in os.listdir(_input_dir_path + '/' + data[i] + '/' + j):
                    path = _input_dir_path + '/' + data[i] + '/' + j + '/' + k   # 某一失真水平的模型
                    if k.endswith(".ply"):
                        file.append(k)
                        if distortion_type == 'random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                            pair_txt_train.write('%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'com&down' ))
                        else:
                            pair_txt_train.write(
                                '%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'noise'))
                       # pair_txt_train.write('%s %s %s %d\n' % (original_model, path, original_model, 0))
                for k in range(len(file)):
                    for h in range(k+1, len(file)):
                        if file[k] > file[h]:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 0,'com&down'))
                            else:
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 0,
                                                                           'noise'))
                        else:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 1,'com&down'))
                            else:
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 1,
                                                                           'noise'))

    # 生成测试文档
    for i in range(int(size * _ratio_for_train), size):
        model_dir = os.listdir(_input_dir_path + '/' + data[i])   # 某一模型文件夹
        original_model = ''
        for j in model_dir:
            path_j = _input_dir_path + '/' + data[i] + '/' + j
            # print(path_j)
            if os.path.isfile(path_j) and j.endswith(".ply"):
                original_model = path_j
        for j in model_dir:
            if os.path.isdir(_input_dir_path + '/' + data[i] + '/' + j):  # 某一噪声类型的文件夹
                distortion_type =  j
                file = []
                for k in os.listdir(_input_dir_path + '/' + data[i] + '/' + j):
                    path = _input_dir_path + '/' + data[i] + '/' + j + '/' + k   # 某一噪声水平的模型
                    if k.endswith(".ply"):
                        file.append(k)
                        if distortion_type == 'random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                            pair_txt_test.write('%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'com&down' ))
                        else:
                            pair_txt_test.write(
                                '%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'noise'))
                for k in range(len(file)):
                    for h in range(k+1, len(file)):
                        if file[k] > file[h]:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 0,'com&down'))
                            else:
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 0,
                                                                           'noise'))
                        else:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 1,'com&down'))
                            else:
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 1,
                                                                           'noise'))


def data_txt_generate2(_input_dir_path, _out_dir_path, _ratio_for_train):
    data = os.listdir(_input_dir_path)
    size = len(data)   # 模型总数
    if not os.path.exists(_out_dir_path):
        os.makedirs(_out_dir_path)
    for num in range(5):  #  交叉验证5次
        # fold = _out_dir_path+'/'+'session'+str(num)
        # if not os.path.exists(fold):
        #     os.makedirs(fold)
        pair_txt_train = open(_out_dir_path+'/'+'rank_pair_train'+str(num)+'.txt', mode="w", encoding="utf-8")
        pair_txt_test = open(_out_dir_path+'/'+'rank_pair_test'+str(num)+'.txt', mode="w", encoding="utf-8")
        single_txt_test = open(_out_dir_path+'/'+'single_test'+str(num)+'.txt', mode="w", encoding="utf-8")
        index = torch.randperm(size)
    # 生成训练文档
        for i in range(0, int(size * _ratio_for_train)):
            model_dir = os.listdir(_input_dir_path + '/' + data[index[i]])   # 某一模型文件夹
            original_model = ''
            for j in model_dir:
                path_j = _input_dir_path + '/' + data[index[i]] + '/' + j
                # print(path_j)
                if os.path.isfile(path_j) and j.endswith(".ply"):
                    original_model = path_j
                    # print(original_model)
            for j in model_dir:
                if os.path.isdir(_input_dir_path + '/' + data[index[i]] + '/' + j):  # 某一失真类型的文件夹
                    distortion_type = j
                    file = []
                    for k in os.listdir(_input_dir_path + '/' + data[index[i]] + '/' + j):
                        path = _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + k   # 某一失真水平的模型
                        if k.endswith(".ply"):
                            file.append(k)
                            if distortion_type == 'random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'com&down' ))
                            else:
                                pair_txt_train.write(
                                    '%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'noise'))
                           # pair_txt_train.write('%s %s %s %d\n' % (original_model, path, original_model, 0))
                    for k in range(len(file)):
                        for h in range(k+1, len(file)):
                            if file[k] > file[h]:
                                if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                    pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 0,'com&down'))
                                else:
                                    pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 0,
                                                                               'noise'))
                            else:
                                if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                    pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 1,'com&down'))
                                else:
                                    pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 1,
                                                                               'noise'))

        # 生成测试文档
        for i in range(int(size * _ratio_for_train), size):
            model_dir = os.listdir(_input_dir_path + '/' + data[index[i]])   # 某一模型文件夹
            original_model = ''
            for j in model_dir:
                path_j = _input_dir_path + '/' + data[index[i]] + '/' + j
                # print(path_j)
                if os.path.isfile(path_j) and j.endswith(".ply"):
                    original_model = path_j
                    single_txt_test.write('%s\n' % (original_model))
            for j in model_dir:
                if os.path.isdir(_input_dir_path + '/' + data[index[i]] + '/' + j):  # 某一噪声类型的文件夹
                    distortion_type =  j
                    file = []
                    for k in os.listdir(_input_dir_path + '/' + data[index[i]] + '/' + j):
                        path = _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + k   # 某一噪声水平的模型
                        if k.endswith(".ply"):
                            file.append(k)
                            single_txt_test.write('%s\n' % (path))
                            if distortion_type == 'random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'com&down' ))
                            else:
                                pair_txt_test.write(
                                    '%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'noise'))
                    for k in range(len(file)):
                        for h in range(k+1, len(file)):
                            if file[k] > file[h]:
                                if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                    pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 0,'com&down'))
                                else:
                                    pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 0,
                                                                               'noise'))
                            else:
                                if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                    pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                        _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 1,'com&down'))
                                else:
                                    pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[k],
                                                                               _input_dir_path + '/' + data[index[i]] + '/' + j + '/' + file[h], 1,
                                                                               'noise'))
        pair_txt_train.close()
        pair_txt_test.close()
        single_txt_test.close()


def data_txt_generate3(_input_dir_path, _out_train_txt, _out_test_txt, _ratio_for_train):
    data = os.listdir(_input_dir_path)
    size = len(data)   # 模型总数
    pair_txt_train = open(_out_train_txt, mode="w", encoding="utf-8")
    pair_txt_test = open(_out_test_txt, mode="w", encoding="utf-8")
    # 生成训练文档
    for i in range(0, int(size * _ratio_for_train)):
        model_dir = os.listdir(_input_dir_path + '/' + data[i])   # 某一模型文件夹
        original_model = ''
        for j in model_dir:
            path_j = _input_dir_path + '/' + data[i] + '/' + j
            # print(path_j)
            if os.path.isfile(path_j) and j.endswith(".ply"):
                original_model = path_j
                # print(original_model)
        for j in model_dir:
            if os.path.isdir(_input_dir_path + '/' + data[i] + '/' + j):  # 某一失真类型的文件夹
                distortion_type = j
                file = []
                for k in os.listdir(_input_dir_path + '/' + data[i] + '/' + j):
                    path = _input_dir_path + '/' + data[i] + '/' + j + '/' + k   # 某一失真水平的模型
                    if k.endswith(".ply"):
                        file.append(k)
                        if distortion_type == 'random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                            pair_txt_train.write('%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'com&down' ))
                        else:
                            pair_txt_train.write(
                                '%s %s %s %d %s\n' % (original_model, original_model, path, 1, 'noise'))
                       # pair_txt_train.write('%s %s %s %d\n' % (original_model, path, original_model, 0))
                for k in range(len(file)):
                    for h in range(k+1, len(file)):
                        if file[k] > file[h]:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 0,'com&down'))
                            else:
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 0,
                                                                           'noise'))
                        else:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 1,'com&down'))
                            else:
                                pair_txt_train.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 1,
                                                                           'noise'))

    # 生成测试文档
    for i in range(int(size * _ratio_for_train), size):
        model_dir = os.listdir(_input_dir_path + '/' + data[i])   # 某一模型文件夹
        original_model = ''
        for j in model_dir:
            path_j = _input_dir_path + '/' + data[i] + '/' + j
            # print(path_j)
            if os.path.isfile(path_j) and j.endswith(".ply"):
                original_model = path_j
        for j in model_dir:
            if os.path.isdir(_input_dir_path + '/' + data[i] + '/' + j):  # 某一噪声类型的文件夹
                distortion_type =  j
                file = []
                for k in os.listdir(_input_dir_path + '/' + data[i] + '/' + j):
                    path = _input_dir_path + '/' + data[i] + '/' + j + '/' + k   # 某一噪声水平的模型
                    if k.endswith(".ply"):
                        file.append(k)
                        if distortion_type == 'random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                            pair_txt_test.write('%s %s %s %d %s\n' % (original_model, original_model, path, 1, distortion_type ))
                        else:
                            pair_txt_test.write(
                                '%s %s %s %d %s\n' % (original_model, original_model, path, 1, distortion_type))
                for k in range(len(file)):
                    for h in range(k+1, len(file)):
                        if file[k] > file[h]:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 0,distortion_type))
                            else:
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 0,
                                                                          distortion_type))
                        else:
                            if distortion_type == 'random' or distortion_type == 'gridAverage' or distortion_type == 'OctreeCom':
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[k],
                                                                    _input_dir_path + '/' + data[i] + '/' + j + '/' + file[h], 1,distortion_type))
                            else:
                                pair_txt_test.write('%s %s %s %d %s\n' % (original_model,
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[k],
                                                                           _input_dir_path + '/' + data[
                                                                               i] + '/' + j + '/' + file[h], 1,
                                                                          distortion_type))




if __name__ == '__main__':
    input_dir_path = './data5'
    out_train_txt = './rank_pair_train.txt'
    out_test_txt = './rank_pair_test.txt'
    data_txt_generate3(input_dir_path, out_train_txt, out_test_txt, 0.8)
    # out_dir_path='./train_and_test'
    # data_txt_generate2(input_dir_path,out_dir_path,0.8)





