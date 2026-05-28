from network import DoubleFusion
import tensorflow as tf
import numpy as np
from data_loader import PointCloudDataset
from tensorflow import keras
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

class MyHinge(tf.keras.Model):
    def __init__(self,margin):
        super(MyHinge,self).__init__()
        self.margin = margin
    def call(self, dist1, dist2, label):
        value = label*(dist1-dist2)-self.margin
        value = tf.nn.relu(value)
        return tf.reduce_mean(value)

class MarginRankingLoss(tf.keras.Model):
    def __init__(self,margin):
        super(MarginRankingLoss,self).__init__()
        self.margin = margin
    def call(self, dist1, dist2, label):
        value = -label*(dist1-dist2)+self.margin
        value = tf.nn.relu(value)
        return tf.reduce_mean(value)


class LossFunc(tf.keras.Model):
    def __init__(self,lamda,margin1,margin2):
        super(LossFunc,self).__init__()
        self.crossloss = keras.losses.BinaryCrossentropy(from_logits=False,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.lowhinge = MarginRankingLoss(margin1)#//no marginloss in keras
        self.highhinge = MyHinge(margin2)
        self.lamda = lamda

    def call(self,dist1,dist2,out,label):
        loss_cross = self.crossloss(label,out)
        loss_low = self.lowhinge(dist1,dist2,label)
        loss_high = self.highhinge(dist1,dist2,label)
        loss_std = loss_cross + self.lamda*(loss_high + loss_low)
        return tf.reduce_mean(loss_std)

def train(trainset,net, criterion, optimizer,epoch,_f_loss=None):   # 使用时
    loss=0
    accs = Ratio()
    for idx, (data1, data2, label) in enumerate(trainset.dataset):
        data1 = tf.transpose(data1,perm=[0, 1, 3, 2])  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = tf.transpose(data2,perm=[0, 1, 3, 2])
        with tf.GradientTape() as tape:
            dist1, dist2, out = net(data1,data2)     # 输出量在GPU上
            loss_net = criterion(label,out) #注意顺序label,out
        grads = tape.gradient(loss_net, net.trainable_weights)
        optimizer.apply_gradients(zip(grads, net.trainable_weights))
        num = correct_num(dist1, dist2)
        loss+=loss_net.numpy().item()
        accs.update(num.item(),data1.shape[0])
        if idx % 50 ==49:
            print("loss:" + str(loss/50))
            if _f_loss is not None:
                _f_loss.write("loss:" + str(loss/50)+"\n")
            loss=0
    print('\ntrain set: epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch, 100. * accs.ratio))


def test(test_set, net, criterion,epoch,_f_acc=None,mode=1):
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

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate,decay_step):
        self.initial_learning_rate = initial_learning_rate
        self.decay_step = decay_step*2

    def __call__(self, step):
        rate = step // self.decay_step
        return self.initial_learning_rate * tf.math.pow(tf.constant(0.8, dtype=tf.float32), tf.cast(rate, dtype=tf.float32))

def main():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    batch_size = 8 #T4=4 V100=8
    lossfile_path = './loss.txt'
    accfile_path = './ave_acc.txt'
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    f_acc = open(accfile_path, 'w', encoding='utf-8')

    net = DoubleFusion()
    x1 = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(batch_size, 64, 3, 516))
    x2 = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(batch_size, 64, 3, 516))
    dist1, dist2, out = net(x1,x2, training_flag=True)

    ckpt = '/userhome/PRL-GQA/TensorFlow/best_7.h5'
    if os.path.exists(ckpt):
        net.load_weights(ckpt)
        print('Successfully loaded '+ckpt)

    epochs = 20
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            train_ds = PointCloudDataset('./rank_pair_train.txt',batch_size=batch_size, is_train=True)
            test_ds = PointCloudDataset('./rank_pair_test.txt',batch_size=1, is_train=False,catergory=True)
            criterion = keras.losses.BinaryCrossentropy(from_logits=False,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            cur_lr = 0.00001
            learning_rate_fn = MyLRSchedule(cur_lr,len(train_ds.dataset))
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        else:
            train_ds = PointCloudDataset('./rank_pair_train.txt',batch_size=batch_size, is_train=True)
            test_ds = PointCloudDataset('./rank_pair_test.txt',batch_size=1, is_train=False,catergory=True)
        train(train_ds, net, criterion, optimizer, epoch, f_loss)
        steps = optimizer.iterations
        print("steps:",steps.numpy())
        lr = learning_rate_fn(steps)
        print('The current learning rate is {:.08f}'.format(lr))
        acc = test(test_ds, net, criterion, epoch, f_acc)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            net.save_weights(f'/userhome/PRL-GQA/TensorFlow/best_{str(epoch)}.h5')
    print('\nbest_epoch: {:d}, best Accuracy: {:.2f}%\n'.format(
        best_epoch, 100. * best_acc))
    print("finish training")
    f_loss.close()
    f_acc.close()

if __name__ == '__main__':
    main()