from network import DoubleFusion
import tensorflow as tf
import numpy as np
from data_loader import PointCloudDataset
from tensorflow import keras
import time
import os

def correct_num(dista, distb):    # 该函数可以换成计算out值>0.5的概率 输入的是tensor
    margin = 0
    pred = dista - distb - margin
    return np.sum(pred.numpy() > 0)*1.0

class Ratio(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.num1=0
        self.num2=0
        self.ratio=0
    def update(self,num1,num2):
        self.num1+=num1
        self.num2+=num2
        self.ratio=self.num1/self.num2

def test(test_set, net, criterion,_f_acc=None,mode=1):
    loss = Ratio()
    accs = Ratio()
    accs_comandown=Ratio()
    accs_noise=Ratio()

    for idx, (data1, data2, label,category) in enumerate(test_set.dataset):
        data1 = tf.transpose(data1, perm=[0, 1, 3, 2])  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = tf.transpose(data2, perm=[0, 1, 3, 2])
        dist1, dist2, out = net(data1,data2, training_flag=False)
        num = correct_num(dist1, dist2)
        loss_net = criterion(label,out)
        print("idx/all: ",idx,"/",len(test_set.dataset))
        loss.update(loss_net.numpy().item()*data1.shape[0], data1.shape[0])
        accs.update(num.item(),data1.shape[0])
        if 'com&down' in category.numpy().item().decode('utf-8'):
            accs_comandown.update(num.item(),data1.shape[0])
        else:
            accs_noise.update(num.item(),data1.shape[0])
    if mode==1:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        loss.ratio, 100. * accs.ratio))
        print('com & down Accuracy: {:.4f}%, noise Accuracy: {:.4f}%\n'.format(
            100. * accs_comandown.ratio, 100. * accs_noise.ratio))
    else:
        print('\ntrain set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
            loss.ratio, 100. * accs.ratio))
    if _f_acc is not None:
        _f_acc.write('Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        loss.ratio, 100. * accs.ratio))
    return accs.ratio

def main():
    start_time = time.time()
    print("start time:",start_time)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    test_ds = PointCloudDataset('./rank_pair_test.txt',batch_size=1, is_train=False,catergory=True)
    accfile_path = './ave_acc_test.txt'
    f_acc = open(accfile_path, 'w', encoding='utf-8')

    net = DoubleFusion()
    x1 = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 64, 3, 516))
    x2 = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 64, 3, 516))
    dist1, dist2, out = net(x1,x2, training_flag=True)

    ckpt = '/userhome/PRL-GQA/TensorFlow/best_7.h5'
    if os.path.exists(ckpt):
        net.load_weights(ckpt)
        print('Successfully loaded '+ckpt)
    criterion = keras.losses.BinaryCrossentropy(from_logits=False,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    acc = test(test_ds, net, criterion, f_acc)
    print("acc:",acc)
    f_acc.close()
    end_time = time.time()
    print("test time(s):",end_time - start_time)

if __name__ == '__main__':
    main()