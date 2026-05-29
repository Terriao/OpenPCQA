import mindspore as ms
import mindspore.nn as nn
import numpy as np
import time
import os

from scipy.stats import pearsonr, spearmanr

from lib.utils import Printl
from models.Res_Models import ResSCNN


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, label):
        y = self._backbone(x)
        print('y', y)
        print('label', label)
        return self._loss_fn(y, label)

    @property
    def backbone_network(self):
        return self._backbone
      
      

class Trainer(object):
    def __init__(self, config, train_loader, test_loader):
        
        self.model = ResSCNN(config.bn_momentum)
        
        if config.phase=='test':
            self.ckpt_path = '/userhome/ResSCNN_CKPT/60.ckpt'
            print('loaded ' + self.ckpt_path)
            ms.load_checkpoint(ckpt_file_name=self.ckpt_path, net=self.model)

        self.config = config
        self.max_epoch = config.max_epoch

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.train_data_size = self.train_loader.get_dataset_size()
        print('训练集大小：', self.train_data_size)
        self.train_loader = self.train_loader.batch(batch_size=8, drop_remainder=False, per_batch_map=None)
        self.num_steps_per_epoch = self.train_loader.get_dataset_size() 
        print('一个epoch的batch数：', self.num_steps_per_epoch)
        
        self.test_loader = self.test_loader.batch(batch_size=1, drop_remainder=False, per_batch_map=None)
        self.test_data_size = self.test_loader.get_dataset_size()
        print('测试集大小：', self.test_data_size)

        self.printl = Printl('/code/ResSCNN-mindspore/print.log')


    def train(self):
        #print(self.model)
        print('>>>>>>>>>>>>>>>> beginning trainning >>>>>>>>>>>>>>>')
        
        self.criterion = ms.nn.HuberLoss()

        self.start_epoch = 0
        
        self.scheduler = ms.nn.ExponentialDecayLR(
            learning_rate=self.config.lr, decay_rate=self.config.exp_gamma, decay_steps=self.num_steps_per_epoch)
        
        if self.config.optimizer == "SGD":
            self.optimizer = ms.nn.SGD(self.model.trainable_params(), learning_rate=self.scheduler, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        
        #self.optimizer = ms.nn.Adam(self.model.trainable_params(), learning_rate=self.scheduler, beta1=0.9, beta2=0.999, weight_decay=self.config.weight_decay)
                 
        loss_net = MyWithLossCell(self.model, self.criterion)
        train_net = nn.TrainOneStepCell(loss_net, self.optimizer)
        
        
        
        for epoch in range(self.start_epoch, self.max_epoch):
            start_time = time.time()
            running_loss = 0.0
            loss_corrected = 0.0
            running_duration = 0.0
            beta = 0.9
            self.model.set_train(True)
            

            for step, (pc, MOSlabel) in enumerate(self.train_loader):
                #print('2:', pc.shape, MOSlabel.shape)
                loss = train_net(pc, MOSlabel.reshape((-1,1)))
                # Statistics -> Loss
                running_loss = beta * running_loss + (1 - beta) * loss.asnumpy()
                loss_corrected = running_loss / (1 - beta ** (step+1))
                # Statictics -> time
                current_time = time.time()
                duration = current_time - start_time
                running_duration = beta * running_duration + (1 - beta) * duration
                duration_corrected = running_duration / (1 - beta ** (step+1))


                lr = self.optimizer.get_lr()
                format_str = '(E:%d, S:%d) [loss = %.4f lr = %.6e] (%.3f sec/batch)'
                print_str = format_str % (epoch, step, loss_corrected, lr, duration_corrected)
                #print(print_str)
                self.printl(print_str)
                start_time = time.time()

            
            ms.save_checkpoint(self.model, '/userhome/ResSCNN_CKPT-2/' + str(epoch)+'.ckpt')
            self._test_epoch()
                
                    

    def _test_epoch(self):
        print('>>>>>>>>>>>>>>>> beginning test >>>>>>>>>>>>>>>')
        
        score_list = np.array([], dtype=float)
        qual_pred_list = np.array([], dtype=float)
        
        self.model.set_train(False)

        for _, (pc, MOSlabel) in enumerate(self.test_loader):
            try:
                score_pred = self.model(pc)
            except ValueError:
                print(ValueError)

            score_list = np.append(score_list, MOSlabel.asnumpy())
            qual_pred_list = np.append(qual_pred_list, score_pred.asnumpy())

        plcc = pearsonr(score_list, qual_pred_list)[0]
        srocc = spearmanr(score_list, qual_pred_list)[0]

        self.printl("PLCC: {:.4f}, SROCC: {:.4f}".format(plcc, srocc))

        return plcc, srocc
        
