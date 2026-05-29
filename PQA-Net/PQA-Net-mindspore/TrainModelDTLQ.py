# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
# [GCC 7.5.0]
# Embedded file name: /userhome/BBBBBBBBB/PQA-Net-mindspore/TrainModelDTLQ.py
# Compiled at: 2023-01-12 16:21:47
# Size of source mod 2**32: 33961 bytes
import scipy.stats, os, time
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.train import Model, LearningRateScheduler
import numpy as np
from ImageQualityDatasetLQSixSeperate import ImageQualityDatasetLQSixSeperate
from MeonLQ import MeonDT
from Gdn import Gdn
from PLCCLoss import PLCCLoss
from VideoTransforms import DenseSpatialCrop_collate
import re, collections, math
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import torch2ms_tools

ms.set_seed(0)

class MyWithLossCell(nn.Cell):

    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, label):
        y = self._backbone(x)
        batch_size = int(y.shape[0])
        y_avg = y.view(batch_size, 1, -1).mean(1)
        self.y_avg = y_avg
        return self._loss_fn(y_avg, label)

    @property
    def backbone_network(self):
        return self._backbone


class Trainer(object):

    def __init__(self, config):
        ms.set_seed(config.seed)
        self.output_channel = config.output_channel
        self.enable_dist_test = True
        self.train_transform = transforms.Compose([vision.CenterCrop(size=235), vision.ToTensor()])
        self.test_transform = lambda stride: transforms.Compose([DenseSpatialCrop_collate(output_size=235, stride=stride)])
        self.val_transform = self.test_transform
        self.train_batch_size = config.batch_size
        self.train_data = ImageQualityDatasetLQSixSeperate(csv_file_dist=(config.train_csv_DT), csv_file_mos=(config.train_csv), root_dir_dist=(config.trainsetDT),
          root_dir_mos=(config.trainset),
          enable_dist=(self.enable_dist_test),
          transform=(self.test_transform),
          train=(config.train))
        
        #ds.config.set_enable_watchdog(False)
        #ms.dataset.config.set_prefetch_size(size=1000)
        #ds.config.set_enable_shared_mem(True)
        #ds.config.set_auto_offload(True)
        #ds.config.set_enable_autotune(True)
        #ds.config.set_prefetch_size(1000)
        self.train_loader = ds.GeneratorDataset((self.train_data), column_names=["image", "score", "disttype", "image_name", "patch_num"], num_parallel_workers=8, shuffle=True, python_multiprocessing=True)
        
#         train_loader = ds.GeneratorDataset(trainset, column_names=["video", "features", "labels", "_"],
#             num_parallel_workers=config.num_workers, shuffle=True, python_multiprocessing=False) #使用多线程
#         train_loader = train_loader.batch(config.train_batch_size, drop_remainder=False)
        
        self.train_data_size = self.train_loader.get_dataset_size()
        print('数据集大小：', self.train_data_size)
        self.train_loader = self.train_loader.batch((self.train_batch_size), drop_remainder=False, per_batch_map=None)
        self.num_steps_per_epoch = self.train_loader.get_dataset_size() 
        print('一个epoch的batch数：', self.num_steps_per_epoch)
        self.writer = SummaryWriter(log_dir=(config.board))
        self.resume = config.resume
        self.train = config.train
        # disable
        self.val_config = {'name':'val_CRIQAV3', 
         'num_workers':1, 
         'input_csv':'/mnt/raid0/CR-IQA-Database-V3/CR-IQA-Database/val_with_pristine_with_dist_cluster.txt', 
         'root_dir':'/mnt/raid0/CR-IQA-Database-V3/CR-IQA-Database/Stage-2-Images/', 
         'transform':self.val_transform, 
         'save_path':'./val_pretrainDT_cluster_CRIQAV3_results/', 
         'val_batch_size':30}
        self.model = MeonDT(output_channel=(self.output_channel))
        self.model_name = type(self.model).__name__
        print(self.model)
        self.crit_dist_type = nn.CrossEntropyLoss()
        self.loss_dt = None
        self.loss = None
        self.train_loss = None
        self.train_dt_loss = None
        self.val_acc = []
        self.val_results = {}
        self.ckpt_path = config.ckpt_path
        self.ckpt = config.ckpt
        self.max_epochs = config.max_epochs
        self.every_eval = config.every_eval
        self.epochs_per_save = config.epochs_per_save
        self.lr_scheduler_name = config.lr_scheduler
        
        if self.lr_scheduler_name == 'StepLR':
            milestone = []
            learning_rates = []
            lr = config.lr
            for i in range(1, int(config.max_epochs / config.decay_interval) + 1):
                milestone.append(config.decay_interval * i)
                learning_rates.append(lr)
                lr = lr * config.decay_ratio
            
            self.lr_dynamic = nn.piecewise_constant_lr(milestone, learning_rates)
            
        if self.lr_scheduler_name == 'CosineAnnealingLR':
            self.lr_dynamic = nn.cosine_decay_lr(min_lr=0.0, max_lr=config.lr,
              total_step=config.max_epochs * self.num_steps_per_epoch,
              step_per_epoch=self.num_steps_per_epoch,
              decay_epoch=config.max_epochs)
            #self.lr_dynamic = nn.CosineDecayLR(min_lr=0.0, max_lr=config.lr, decay_steps=config.max_epochs * self.num_steps_per_epoch)
        else:
            raise Exception('Wrong lr_scheduler_name')
        
        #print('init lr_dynamic:', len(self.lr_dynamic), self.lr_dynamic[30*317:(30*317+1000)])
        self.optimizer = nn.Adam(params=(self.model.trainable_params()), learning_rate=config.lr)
        
        if self.resume or not self.train:
            if self.ckpt:
                ckpt = os.path.join(self.ckpt_path, self.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)
            self.last_epoch = self.current_epoch
            print(self.last_epoch)
            #last_lr = self.optimizer.get_lr().asnumpy().tolist()
            last_step = (self.last_epoch+1)*self.num_steps_per_epoch
            #self.optimizer = nn.Adam(params=(self.model.trainable_params()), learning_rate=self.lr_dynamic[last_step:])
            #print('resume lr_dynamic:', len(self.lr_dynamic[last_step:]), self.lr_dynamic)
            self.global_step = ms.Tensor([last_step + 1])
            #self.optimizer.lr = last_lr
        else:
            self.global_step = ms.Tensor([0])
            self.last_epoch = -1

    def fit(self):
        ### 定义移到外面，否则内存一直涨
        self.loss_net = MyWithLossCell(self.model, self.crit_dist_type)
        self.train_net = nn.TrainOneStepCell(self.loss_net, self.optimizer)
        for epoch in range(self.last_epoch + 1, self.max_epochs):
            self._train_single_epoch(epoch)

    def _evaluateImage_denseCrop(self, test_config):
        if test_config is None:
            return (None, None)
        elif self.train:
            self.test_data = ImageQualityDatasetLQSixSeperate(csv_file_dist=(test_config['input_mos_csv']), csv_file_mos=(test_config['input_dis_csv']),
              root_dir_dist=(test_config['root_dist_dir']),
              root_dir_mos=(test_config['root_mos_dir']),
              enable_dist=(self.enable_dist_test),
              transform=(self.test_transform(128)),
              train=True)
        else:
            self.test_data = ImageQualityDatasetLQSixSeperate(csv_file_dist=(test_config['input_csv']), csv_file_mos=(test_config['input_csv']),
              root_dir_dist=(test_config['root_dir']),
              root_dir_mos=(test_config['root_dir']),
              enable_dist=(self.enable_dist_test),
              transform=(self.test_transform(128)),
              train=(self.train))
        self.test_loader = ds.GeneratorDataset(self.test_data, column_names=["image", "score", "disttype", "image_name", "patch_num"], num_parallel_workers=1, shuffle=False)
        length = self.test_loader.get_dataset_size()
        self.test_loader = self.test_loader.batch((test_config['test_batch_size']), drop_remainder=False, per_batch_map=None)
        print('-------------->test length:', length)
        image_name_list = []
        disttype_list = np.zeros([length], dtype=(np.int))
        disttype_predict_list = np.zeros([length, self.output_channel])
        batch_size = test_config['test_batch_size']
        for counter, sample_batched in enumerate(self.test_loader, 0):
            start_time = time.time()
            image_batch, score_batch, disttype_batch, name_batch, patch_num_batch = sample_batched
            image = image_batch
            print('image.shape:', image.shape)
            disttype = disttype_batch.asnumpy()
            self.model.set_train(False)
            disttype_predict = self.model(image)
            disttype_predict = disttype_predict.asnumpy()
            patch_counter = 0
            for i in range(len(patch_num_batch)):
                disttype_predict_list[counter * batch_size + i] = np.mean((disttype_predict[patch_counter:patch_counter + patch_num_batch[i], :]), axis=0)
                patch_counter += patch_num_batch[i]
            
            disttype_list[counter * batch_size:(counter + 1) * batch_size] = disttype
            image_name_list += name_batch
            stop_time = time.time()
            samples_per_sec = batch_size / (stop_time - start_time)
            print(batch_size, counter * batch_size, '/', length, '\tSamples/Sec', samples_per_sec)

        max_idxs = np.argmax(disttype_predict_list, axis=1)
        acc = np.sum(np.equal(max_idxs, disttype_list)) / len(disttype_list)
        confuse_m = confusion_matrix(disttype_list, max_idxs)
        print('Confusion matrix: ')
        print(confuse_m)
        self.writer.add_scalar('TestAcc', acc, self.current_epoch * self.num_steps_per_epoch)
        print('Acc: ', acc)
        if test_config['save_path'] is not None:
            save_path = os.path.join(test_config['save_path'], self.model_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, self.model_name + '_' + str(self.current_epoch) + '_' + test_config['name'] + '.ckpt')
            result = {'db_name':test_config['name'],  'model_name':self.model_name, 
             'image_names':image_name_list, 
             'acc':acc}
        test_result_file = os.path.join(save_path, self.model_name + '_' + str(self.current_epoch) + '_' + test_config['name'] + '.txt')
        np.savetxt(test_result_file, (np.column_stack([image_name_list, max_idxs, disttype_list])), fmt='%s', delimiter=',')
        return acc

    def _train_single_epoch(self, epoch):
        self.current_epoch = epoch
        local_counter = epoch * self.num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss
        loss_corrected = 0.0
        running_duration = 0.0
        
        self.model.set_train()
        
        for step, sample_batched in enumerate(self.train_loader, 0):
            #print('global_step:',self.global_step)
            ms.ops.assign(self.optimizer.learning_rate, self.lr_dynamic[self.global_step])
            #print('current_lr:', self.optimizer.learning_rate.data.asnumpy())
            self.global_step = self.global_step + 1
            
            images_batch, score_batch, disttype_batch, _, _ = sample_batched
            image = images_batch
            disttype = disttype_batch
            #print('-----------》image：', image.shape)
            #print('-----------》disttype：', disttype.shape, disttype)
            
            self.loss = self.train_net(image, disttype)
            y_avg = self.loss_net.y_avg
            max_idxs, _ = ms.ops.max(y_avg, axis=1)
            train_acc = np.sum(np.equal(max_idxs.asnumpy(), disttype_batch.asnumpy())) / len(disttype_batch)
            self.writer.add_scalar('TrainAcc', train_acc, local_counter)
            self._gdn_param_proc()
            running_loss = beta * running_loss + (1 - beta) * self.loss.asnumpy().tolist()
            loss_corrected = running_loss / (1 - beta ** local_counter)
            self.writer.add_scalar('TrainLoss', self.loss.asnumpy().tolist(), local_counter)
            lr = self.optimizer.get_lr()
            self.writer.add_scalar('lr', lr.asnumpy(), local_counter)
            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = '(E:%d, S:%d) [loss_dt = %.4f, total loss = %.4f, acc = %.4f, lr = %.6e] (%.1f samples/sec; %.3f sec/batch)'
            print_str = format_str % (epoch, step, self.loss.asnumpy().tolist(), loss_corrected, train_acc, lr, examples_per_sec, duration_corrected)
            print(print_str)
            local_counter += 1
            start_time = time.time()
            #self.train_loss.append(loss_corrected)
            #self.train_dt_loss.append(self.loss.asnumpy().tolist())
            
            
#             self.train_loss = loss_corrected
#             model_name = '{}-{:0>5d}.ckpt'.format(self.model_name, epoch)
#             model_name = os.path.join(self.ckpt_path, model_name)
#             self._save_checkpoint(self.model, {'epoch': epoch, 'train_loss':self.train_loss}, model_name)
        
        self.train_loss = loss_corrected
        # disable
        if (epoch + 1) % self.every_eval == 0:
            val_results = self.eval_test(self.val_config)
            out_str = 'Epoch {}'.format(epoch)
            out_str += ' Validating '
            for db_name in val_results:
                if db_name in self.val_results:
                    self.val_results[db_name].append(val_results[db_name])
                else:
                    self.val_results[db_name] = [
                     val_results[db_name]]
                result = val_results[db_name]
                out_str += '\n' + db_name + '\tAcc: ' + str(result[0])
            
            print(out_str)
            self.val_acc.append(result[0])

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.ckpt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint(self.model, {'epoch': epoch, 'train_loss':self.train_loss}, model_name)

    def eval_test(self, *args):
        """
        Evaluate distortion type accuracy and quality prediction SRCC for all test databases in args.
        All results are saved in self.test_results.
        :param args: dicts of configurations for evaluating databases
        :return results: a dictionary containing classification accuracies and SRCCs for all eval databases
        """
        results = {}
        for val_config in args:
            db_name = val_config['name']
            print('Evaluating: {} database'.format(db_name))
            acc = self._evaluateImage_denseCrop(val_config)
            results[db_name] = [
             acc]
        
        return results

    def _gdn_param_proc(self):
        for name, cell in self.model.cells_and_names():
            if isinstance(cell, Gdn):
                cell.beta.set_data(ms.ops.clamp((cell.beta.value()), min=2e-10), slice_shape=True)
                cell.gamma.set_data(ms.ops.clamp((cell.gamma.value()), min=2e-10), slice_shape=True)
                cell.gamma = cell.gamma.set_data(((cell.gamma + cell.gamma.transpose((1,
                                                                                      0))) / 2), slice_shape=True)

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint_dict = ms.load_checkpoint(ckpt)
            self.current_epoch = checkpoint_dict['epoch'].asnumpy().tolist()
            self.train_loss = checkpoint_dict['train_loss'].asnumpy().tolist()  # last train_loss
            del checkpoint_dict['epoch']
            del checkpoint_dict['train_loss']
            #ms.load_checkpoint((checkpoint_dict.pop('epoch')), net=(self.model))
            ms.load_param_into_net(net=self.model, parameter_dict=checkpoint_dict)
            print("[*] loaded checkpoint '{}' (epoch {})".format(ckpt, self.current_epoch))
        else:
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    @staticmethod
    def _save_checkpoint(model, state_dict, filename='checkpoint.ckpt'):
        ms.save_checkpoint(save_obj=model, ckpt_file_name=filename, append_dict=state_dict)