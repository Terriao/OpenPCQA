import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import tensorflow as tf
import argparse
import numpy as np

from utils import fit_dt, Printl
from PQAmodels import MeonDT
from PQADataset import ImageDataset

def main(args):
    model = MeonDT(args.output_channel)

    if args.resume:
        imgs = tf.random.uniform((1, 6, 235, 235, 3), minval=0, maxval=255)
        model(imgs)
        model.load_weights(os.path.join(args.ckpt_dir, args.ckpt), by_name=True, skip_mismatch=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    printl = Printl(os.path.join(args.log_dir, "Distortion-WPC-2.txt"))

    train_data = ImageDataset(
        csv_file_dist=args.train_csv_DT,
        root_dir_dist=args.trainsetDT,
        batch_size=args.batch_size,
        shuffle=2000,
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
        bds = list(np.arange(0, args.max_epochs, args.decay_interval))
        vals = [(args.decay_ratio)**i for i in range(len(bds))]
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=bds, values=vals)
    else:
        raise Exception("Wrong lr_scheduler_name")

    fit_dt(model, train_data=train_data, test_data=val_data, criterion=criterion, optimizer=optimizer, scheduler=scheduler, args=args, printl=printl)

    return 0



if __name__=="__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--resume", type=bool, default=False)

    parser.add_argument("--trainsetDT", type=str, default="/public/DATA/lhh/WPCSD/distortion")
    parser.add_argument("--train_csv_DT", type=str, default="./data/PCMeon2DelDMOSSameTrainbcmp_dist.txt")
    # parser.add_argument("--testsetDT", type=str, default="/media/abc/pcqa_databases/WPCSD/distortion/")
    parser.add_argument("--test_csv_DT", type=str, default="./data/PCMeon2DelDMOSSameTestbcmp_mos.txt")

    parser.add_argument("--output_channel", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_interval", type=int, default=8)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--every_eval", type=int, default=5)
    parser.add_argument("--epoch_save", type=int, default=5) # disable
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str, metavar='PATH',help='path to checkpoints')
    parser.add_argument("--ckpt", default="MeonDT-00029.h5", type=str)

    parser.add_argument("--log_dir", default="./logs")

    parser.add_argument('--lr_scheduler', default="CosineAnnealingLR", type=str, help='CosineAnnealingLR or StepLR')

    cfgs = parser.parse_args()

    np.random.seed(cfgs.seed)
    tf.random.set_seed(cfgs.seed)

    main(cfgs)