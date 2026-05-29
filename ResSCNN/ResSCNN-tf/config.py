import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--resume', type=bool, default=True)
parser.add_argument('--ckpt_dir', type=str, default="./checkpoints")
parser.add_argument('--resume_ckpt', type=str, default="ResSCNN-00041_LS-PCQA.h5")

parser.add_argument('--norm_size', default=1800)
parser.add_argument('--voxel_size', default=5)

parser.add_argument('--use_random_scale', default=False)
parser.add_argument('--min_scale', type=float, default=0.8)
parser.add_argument('--max_scale', type=float, default=1.0)
parser.add_argument('--use_random_rotation', default=True)
parser.add_argument('--rotation_range', type=float, default=360)

# Training set and Testing set
parser.add_argument('--train_file', type=str, default='./config/train.csv', help='file name and MOS for training set')
parser.add_argument('--test_file', type=str, default='./config/test.csv', help='file name and MOS for testing set')
parser.add_argument('--file_dir', type=str, default='/public/DATA/lhh/LS-PCQA/samples_with_MOS')
parser.add_argument('--train_path', type=str, default='./config/path.xlsx', help='file name and file path for training set')
parser.add_argument('--test_path', type=str, default='./config/path.xlsx', help='file name and file path for testing set')
parser.add_argument('--every_eval', type=int, default=1)


# Optimizer arguments
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--bn_momentum', type=float, default=0.95)
parser.add_argument('--exp_gamma', type=float, default=0.99)

# Should not be changed
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)


parser.add_argument('--log_file', type=str, default="./logs/ls-pcqa_log2.txt")


def get_config():
    config = parser.parse_args()

    return config