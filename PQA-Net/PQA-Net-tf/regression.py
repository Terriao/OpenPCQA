import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import tensorflow as tf
import argparse
import numpy as np

from utils import Printl, fit
from PQADataset import ImageDataset
from PLCCloss import PLCCloss
from PQAmodels import Meon

def main(args):
    model = Meon(args.output_channel)

    if args.pretrainDT:
        imgs = tf.random.uniform((1, 6, 235, 235, 3), minval=0, maxval=255)
        model(imgs)
        model.load_weights(os.path.join(args.ckpt_dir, args.ckpt), by_name=True, skip_mismatch=True)


    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    crit_dt = lambda x,y: tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(x, y, from_logits=True))
    crit_qp = lambda x,y: PLCCloss(x, y)
    printl = Printl(os.path.join(args.log_dir, "Regression_15_3.txt"))

    train_data = ImageDataset(
        csv_file_dist=args.train_csv_DT,
        root_dir_dist=args.trainsetDT,
        batch_size=args.batch_size,
        shuffle=1000,
        transform=True,
        epoch=args.max_epochs
    )

    val_data = ImageDataset(
        csv_file_dist=args.test_csv_DT,
        root_dir_dist=args.trainsetDT,
        batch_size=args.batch_size,
        shuffle=0,
        transform=True,
        epoch=1
    )

    if args.lr_scheduler == "CosineAnnealingLR":
        scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.lr,
                                                            decay_steps=args.max_epochs * len(train_data))
    elif args.lr_scheduler == "StepLR":
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, decay_steps=args.decay_interval, decay_rate=args.decay_ratio)
    else:
        raise Exception("Wrong lr_scheduler_name")

    fit(model, train_data, val_data, crit_dt, crit_qp, optimizer, scheduler, args, printl)



if __name__=="__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--pretrainDT", type=bool, default=True)

    parser.add_argument("--trainsetDT", type=str, default="/public/DATA/lhh/WPCSD/distortion")
    parser.add_argument("--train_csv_DT", type=str, default="./data/PCMeon2DelDMOSSameTrainbcmp_dist.txt")
    # parser.add_argument("--testsetDT", type=str, default="/media/abc/pcqa_databases/WPCSD/distortion/")
    parser.add_argument("--test_csv_DT", type=str, default="./data/PCMeon2DelDMOSSameTestbcmp_mos.txt")

    parser.add_argument("--lamb", type=float, default=-15)

    parser.add_argument("--output_channel", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--decay_interval", type=int, default=10)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--every_eval", type=int, default=1)  # disable
    parser.add_argument("--epoch_save", type=int, default=9999)
    parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, metavar='PATH',help='path to checkpoints')
    parser.add_argument('--ckpt', default="MeonDT-00004_WPC_resume.h5", type=str, help='name of the checkpoint to load')

    parser.add_argument("--log_dir", default="./logs")

    parser.add_argument('--lr_scheduler', default="CosineAnnealingLR", type=str, help='CosineAnnealingLR or StepLR')

    cfgs = parser.parse_args()

    np.random.seed(cfgs.seed)
    tf.random.set_seed(cfgs.seed)

    main(cfgs)