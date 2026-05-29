import tensorflow as tf
import numpy as np
import time
import os

from scipy.stats import pearsonr, spearmanr, kendalltau

from lib.utils import Printl
from models.Res_Models import ResSCNN


class Trainer(object):
    def __init__(self, config, train_loader, test_loader):

        self.model = ResSCNN(config.bn_momentum)

        if config.resume:
            x = tf.random.uniform((1, 400, 400, 400, 3))
            self.model(x)
            self.model.load_weights(os.path.join(config.ckpt_dir, config.resume_ckpt), by_name=True, skip_mismatch=True)

        self.config = config
        self.max_epoch = config.max_epoch

        self.train_loader = train_loader
        self.test_loader = test_loader

        if config.optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.experimental.SGD(
                learning_rate=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )

        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.lr,
            decay_steps=1,
            decay_rate=config.exp_gamma
        )

        self.criterion = tf.keras.losses.Huber()

        self.start_epoch = 1
        self.every_eval = config.every_eval

        self.printl = Printl(config.log_file)


    def train(self):

        self.current_epoch = self.start_epoch

        loader = self.train_loader
        epoch_len = len(loader)

        start_time = time.time()
        running_loss = 0.0
        loss_corrected = 0.0
        running_duration = 0.0
        beta = 0.9

        best_srocc = 0
        for step, (pc, MOSlabel) in enumerate(loader.dataset):
            with tf.GradientTape() as tape:
                score_pred = self.model(pc)
                loss = self.criterion(MOSlabel, score_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Statistics -> Loss
            running_loss = beta * running_loss + (1 - beta) * loss.numpy()
            loss_corrected = running_loss / (1 - beta ** (step+1))
            # Statictics -> time
            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** (step+1))
            examples_per_sec = loader.batch_size / duration_corrected

            lr = self.optimizer.lr
            format_str = '(E:%d, S:%d) [loss = %.4f lr = %.6e] (%.3f samples/sec; %.3f sec/batch)'
            print_str = format_str % (self.current_epoch, step-((self.current_epoch-1)*epoch_len), loss_corrected, lr, examples_per_sec, duration_corrected)
            self.printl(print_str)
            start_time = time.time()

            if (step + 1) % epoch_len == 0:
                if (self.current_epoch) % self.every_eval == 0:
                    plcc, srocc, rmse = self._test_epoch()
                    if srocc > best_srocc:
                        best_srocc = srocc
                        self.save_model_weight(self.config.ckpt_dir)
                        self.printl("Better model saved!")

                self.optimizer.lr = self.scheduler(self.current_epoch-1)

                self.current_epoch += 1

    def _test_epoch(self):

        loader = self.test_loader

        score_list = np.array([], dtype=float)
        qual_pred_list = np.array([], dtype=float)

        for _, (pc, MOSlabel) in enumerate(loader.dataset):
            score_pred = self.model(pc)

            score_list = np.append(score_list, MOSlabel.numpy())
            qual_pred_list = np.append(qual_pred_list, score_pred.numpy())

        plcc = pearsonr(score_list, qual_pred_list)[0]
        srocc = spearmanr(score_list, qual_pred_list)[0]
        krocc = kendalltau(score_list, qual_pred_list)[0]
        rmse = np.sqrt(np.mean((score_list-qual_pred_list)**2))

        self.printl("PLCC: {:.4f}, SROCC: {:.4f} KROCC: {:.4f} RMSE: {:.4f}".format(plcc, srocc, krocc, rmse))

        return plcc, srocc, rmse

    def save_model_weight(self, ckpt_dir):
        model_name = "{}-{:0>5d}_WPC.h5".format(type(self.model).__name__, self.current_epoch)
        model_name = os.path.join(ckpt_dir, model_name)
        self.model.save_weights(model_name)

