import tensorflow as tf
import numpy as np
import time
import os
from scipy.stats import pearsonr, spearmanr


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


def save_model_weight(model, ckpt_dir, epoch):
    model_name = "{}-{:0>5d}_WPC_resume.h5".format(type(model).__name__, epoch)
    model_name = os.path.join(ckpt_dir, model_name)
    model.save_weights(model_name)


def fit_dt(model, train_data, test_data, criterion, optimizer, scheduler, args, printl):
    
    epoch, epoch_len = 0, len(train_data)

    start_time = time.time()
    running_loss = 0.0
    loss_corrected = 0.0
    running_duration = 0.0
    beta = 0.9

    for step, (imgs, _, dis_type) in enumerate(train_data.dataset):
        # Train 
        with tf.GradientTape() as tape:
            dist_pred = model(imgs)
            loss = criterion(dis_type, dist_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ###########

        if args.lr_scheduler == "CosineAnnealingLR":
            optimizer.lr = scheduler(step)

        # Statistics -> Loss
        running_loss = beta * running_loss + (1 - beta) * loss.numpy()
        loss_corrected = running_loss / (1 - beta ** (step+1))
        # Statictics -> time
        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** (step+1))
        examples_per_sec = train_data.batch_size / duration_corrected

        train_acc = np.sum(np.equal(dis_type.numpy(), np.argmax(dist_pred.numpy(), -1))) / train_data.batch_size
        lr = optimizer.lr
        format_str = '(E:%d, S:%d) [loss_dt = %.4f, total loss = %.4f, acc = %.4f, lr = %.6e] (%.1f samples/sec; %.3f sec/batch)'
        print_str = format_str % (epoch, step-(epoch*epoch_len), loss.numpy(), loss_corrected, train_acc, lr, examples_per_sec, duration_corrected)
        printl(print_str)
        start_time = time.time()
        ###########

        if (step+1) % epoch_len == 0:
            if args.lr_scheduler == "StepLR":
                optimizer.lr = scheduler(epoch)

            if (epoch+1) % args.every_eval == 0:
                acc = validate_dt(model, test_data, printl)

            if (epoch+1) % args.epoch_save == 0:
                save_model_weight(model, args.ckpt_dir, epoch)

            epoch += 1


def validate_dt(model, test_data, printl):
    dist_list = np.array([], dtype=int)
    dist_pred_list = np.array([], dtype=int)
    for step, (imgs, _, dis_type) in enumerate(test_data.dataset):
        dist_pred = model(imgs)
        dist_list = np.append(dist_list, dis_type.numpy())
        dist_pred_list = np.append(dist_pred_list, np.argmax(dist_pred.numpy(), axis=-1))

    acc = np.sum(np.equal(dist_list, dist_pred_list)) / len(dist_list)
    printl("Acc: {:.6f}".format(acc))

    return acc


def fit(model, train_data, test_data, crit_dt, crit_qp, optimizer, scheduler, args, printl):
    epoch, epoch_len = 0, len(train_data)

    start_time = time.time()
    running_loss = 0.0
    loss_corrected = 0.0
    running_duration = 0.0
    beta = 0.9

    lamb = args.lamb

    for step, (imgs, score, dis_type) in enumerate(train_data.dataset):
        # Train

        with tf.GradientTape() as tape:
            dist_pred, qual_pred = model(imgs)
            loss_dt = crit_dt(dis_type, dist_pred)
            loss_qp = crit_qp(score, qual_pred)
            loss = loss_dt + lamb*loss_qp
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ###########

        if args.lr_scheduler == "CosineAnnealingLR":
            optimizer.lr = scheduler(step)

        # Statistics -> Loss
        running_loss = beta * running_loss + (1 - beta) * loss.numpy()
        loss_corrected = running_loss / (1 - beta ** (step+1))
        # Statictics -> time
        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** (step+1))
        examples_per_sec = train_data.batch_size / duration_corrected

        train_acc = np.sum(np.equal(dis_type.numpy(), np.argmax(dist_pred.numpy(), -1))) / train_data.batch_size
        lr = optimizer.lr
        format_str = '(E:%d, S:%d) [loss_dt = %.4f, loss_qp = %.4f, total loss = %.4f, acc = %.4f, lr = %.6e] (%.1f samples/sec; %.3f sec/batch)'
        print_str = format_str % (epoch, step-(epoch*epoch_len), loss_dt.numpy(), loss_qp.numpy(), loss_corrected, train_acc, lr, examples_per_sec, duration_corrected)
        printl(print_str)
        start_time = time.time()
        ###########

        if (step+1) % epoch_len == 0:
            if args.lr_scheduler == "StepLR":
                optimizer.lr = scheduler(epoch)

            if (epoch+1) % args.every_eval == 0:
                acc = validate(model, test_data, printl)

            if (epoch+1) % args.epoch_save == 0:
                save_model_weight(model, args.ckpt_dir, epoch)

            epoch += 1


def validate(model, test_data, printl):
    dist_list = np.array([], dtype=int)
    dist_pred_list = np.array([], dtype=int)

    score_list = np.array([], dtype=float)
    qual_pred_list = np.array([], dtype=float)

    for step, (imgs, score, dis_type) in enumerate(test_data.dataset):
        dist_pred, qual_pred = model(imgs)
        dist_list = np.append(dist_list, dis_type.numpy())
        dist_pred_list = np.append(dist_pred_list, np.argmax(dist_pred.numpy(), axis=-1))

        score_list = np.append(score_list, score.numpy())
        qual_pred_list = np.append(qual_pred_list, qual_pred.numpy())

    acc = np.sum(np.equal(dist_list, dist_pred_list)) / len(dist_list)
    plcc = pearsonr(score_list, qual_pred_list)[0]
    srocc = spearmanr(score_list, qual_pred_list)[0]

    printl("PLCC: {:.4f}, SROCC: {:.4f}, Acc: {:.4f}".format(plcc, srocc, acc))

    return plcc, srocc, acc