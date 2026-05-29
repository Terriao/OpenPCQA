import argparse
import TrainModelLQ
import mindspore
import time
import os


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--pretrainDT", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
 #   parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--seed", type=int, default=2019)

 #   parser.add_argument("--trainsetDT", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/trainset/")
    parser.add_argument("--trainsetDT", type=str, default="/dataset/distortion/")
    parser.add_argument("--train_csv_DT", type=str,
                            default="./label/PCMeon2DelDMOSSameTrainbcmp_dist.txt")
 #   parser.add_argument("--trainset", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/trainset/")
    parser.add_argument("--trainset", type=str, default="/dataset/distortion/")
    parser.add_argument("--train_csv", type=str,
                        default="./label/PCMeon2DelDMOSSameTrainbcmp_mos.txt")
    parser.add_argument("--output_channel", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=30)  # 不能为1否则 loss为nan
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-2)  #1e-4
    parser.add_argument("--decay_interval", type=int, default=40) #10
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--every_eval", type=int, default=99999)  # disable
    parser.add_argument("--epochs_per_save", type=int, default=1)
   # parser.add_argument('--ckpt_path', default='./checkpoint_pretrian4bgpsMC/', type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--board', default="./board_pretrain4bgps", type=str, help='tensorboardX log file path')
    parser.add_argument('--lr_scheduler', default="StepLR", type=str, help='CosineAnnealingLR or StepLR')
    # parser.add_argument('--DT_ckpt_path', default='/home/qi/QiLiu/code/MEONCode/MEONLQModelChange/checkpoint_pretrain4DTbgpsMC/', type=str, metavar='PATH', help='path to DT checkpoints')
    
    parser.add_argument('--ckpt_path', default='/userhome/PQA-Net/ms/LQ_ckpt-15/', type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default="Meon-00079.ckpt", type=str, help='name of the checkpoint to load')
    parser.add_argument('--DT_ckpt_path', default='/userhome/PQA-Net/ms/DT_ckpt/', type=str, metavar='PATH', help='path to DT checkpoints')
    parser.add_argument('--DT_ckpt', default="MeonDT-00079.ckpt", type=str, help='name of the DT checkpoint to load')
    
    return parser.parse_args()


def main(cfg):
    t = TrainModelLQ.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        # show_training_curve(cfg.checkpoint)
        start_time = time.time()
        dataset_root = "/dataset/distortion/"
        save_root = "./PC_test2change_results/"
        num_workers = 4
        t.enable_dist_test = True ###need to change it to True to have a try

        test_results = t.eval_test(
        {
            # "name": "test_lr3",
#            "name": "train_lr3",
            "name": "train_lr3_2change",
            "num_workers": num_workers,
            "input_csv": os.path.join("./label", 'PCMeon2DelDMOSSameTestbcmp_mos.txt'),
            "root_dir": dataset_root,
            #"save_path": os.path.join(save_root, "test_lr3"),
            "save_path": os.path.join(save_root, "test_2Change_lr3"),
            "test_batch_size": 1},
        )

        current_time = time.time()
        print("Total time: {:.4f}".format(current_time - start_time))
        for db_name in test_results:
            result = test_results[db_name]
            if t.enable_dist_test:
                out_str = '{}: SRCC {:.4f}, PLCC {:.4f}, Acc {:.4f};'.format(db_name, result[0], result[1], result[2])
            else:
                out_str = '{}: SRCC {:.4f}, PLCC {:.4f}'.format(db_name, result[0], result[1])
            print(out_str)


if __name__ == "__main__":
    config = parse_config()
    main(config)
