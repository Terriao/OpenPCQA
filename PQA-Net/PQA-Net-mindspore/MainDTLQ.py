import argparse
import TrainModelDTLQ
import mindspore as ms
import time
import os
#ms.set_context(device_target='GPU', mode=ms.GRAPH_MODE)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
#    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--resume", type=bool, default=False)

 #   parser.add_argument("--trainsetDT", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/trainset/")
    parser.add_argument("--trainsetDT", type=str, default="/dataset/distortion/")
    parser.add_argument("--train_csv_DT", type=str,
                            default="./label/PCMeon2DelDMOSSameTrainbcmp_dist.txt")
 #   parser.add_argument("--trainset", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/trainset/")
    parser.add_argument("--trainset", type=str, default="/dataset/distortion/")
    parser.add_argument("--train_csv", type=str,
                        default="./label/PCMeon2DelDMOSSameTrainbcmp_mos.txt")
#     parser.add_argument("--train_csv", type=str,
#                             default="/home/qi/QiLiu/code/MEONCode/MEONLQ/PCMeon2DelMOSSame_train_mos.txt")
    parser.add_argument("--output_channel", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_interval", type=int, default=8)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--every_eval", type=int, default=99999)  # disable
    parser.add_argument("--epochs_per_save", type=int, default=1)
    parser.add_argument('--ckpt_path', default='/userhome/PQA-Net/ms/DT_ckpt/', type=str, metavar='PATH',help='path to checkpoints')
  #  parser.add_argument('--ckpt_path', default='./checkpoint_pretrain4DTbgpsMC_dropout_maxview/', type=str, metavar='PATH',
  #                      help='path to checkpoints')
    parser.add_argument('--ckpt', default="MeonDT-00039.ckpt", type=str, help='name of the checkpoint to load')
    parser.add_argument('--board', default="./board_pretrainDT6Sep_lr3", type=str, help='tensorboardX log file path')
#    parser.add_argument('--board', default="./board_pretrainDT6bgpsMC_maxview", type=str, help='tensorboardX log file path')
    parser.add_argument('--lr_scheduler', default="CosineAnnealingLR", type=str, help='CosineAnnealingLR or StepLR')

    return parser.parse_args()


def main(cfg):
    t = TrainModelDTLQ.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        # show_training_curve(cfg.checkpoint)
        start_time = time.time()
        dataset_root = "./dataset/"
        save_root = "./PC_test2Delchange_results/"
#        dataset_root = "/home/qi/QiLiu/code/MEONCode/MEONLQ/trainset/"
        # save_root = "/home/qi/QiLiu/code/MEONCode/MEONLQ/PC_test4D_DT_train_results/"
        num_workers = 1
        t.enable_dist_test = True

        test_results = t.eval_test(
        {
            # "name": "train_lr3_2change_DT_addpoints",
            "name": "test_lr3_2Delchange_DT_addpoints",
            "num_workers": num_workers,
            "input_csv": os.path.join(dataset_root, 'PCMeon2DelDMOSSameTestbcmp_dist.txt'),
            "root_dir": dataset_root,
            "save_path": os.path.join(save_root, "PC_DTTest2DelChange"),
            "test_batch_size": 1},
        )

        current_time = time.time()
        print("Total time: {:.4f}".format(current_time - start_time))
        for db_name in test_results:
            result = test_results[db_name]
            out_str = '{Acc {:.4f};}'.format(result[0])
 #           out_str = '{Acc {:.4f}}'.format(db_name, result[0])
            print(out_str)


if __name__ == "__main__":
    config = parse_config()
    main(config)
