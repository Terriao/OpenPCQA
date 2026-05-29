import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs', help='path for saving the checkpoints')

trainer_arg = add_argument_group('Trainer')
# Should not be changed
trainer_arg.add_argument('--batch_size', type=int, default=1)
trainer_arg.add_argument('--test_batch_size', type=int, default=1)


# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--test_phase', type=str, default="test")
trainer_arg.add_argument('--phase', type=str, default="train")

trainer_arg.add_argument('--stat_freq', type=int, default=100)
trainer_arg.add_argument('--test_eval', type=str2bool, default=True)
trainer_arg.add_argument('--test_max_iter', type=int, default=0, help='0 for no limitation of the number of testing samples')
trainer_arg.add_argument('--test_epoch_freq', type=int, default=1)

# Training set and Testing set
# trainer_arg.add_argument('--train_file', type=str, default='./config/wpc20_train.txt', help='file name and MOS for training set')
# trainer_arg.add_argument('--test_file', type=str, default='./config/wpc20_test.txt', help='file name and MOS for testing set')
# trainer_arg.add_argument('--file_path', type=str, default='/userhome/distorted_PCs', help='file name and file path for training set')

trainer_arg.add_argument('--train_file', type=str, default='./config/train.xlsx', help='file name and MOS for training set')
trainer_arg.add_argument('--test_file', type=str, default='./config/test.xlsx', help='file name and MOS for testing set')
trainer_arg.add_argument('--file_path', type=str, default='/userhome/LS-PCQA/Distortion', help='file name and file path for training set')


# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-3)
opt_arg.add_argument('--momentum', type=float, default=0.8)

opt_arg.add_argument('--weight_decay', type=float, default=1e-3)
opt_arg.add_argument('--iter_size', type=int, default=8, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--resume', type=str, default=None, help='path for loading the checkpoint')
misc_arg.add_argument('--train_num_thread', type=int, default=8)
misc_arg.add_argument('--test_num_thread', type=int, default=8)


# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--voxel_size', type=float, default=5, help='voxelization size for sparse convolution')


def get_config():
  args = parser.parse_args()
  return args
