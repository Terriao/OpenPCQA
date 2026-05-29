import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
import numpy as np

from config import get_config
from lib.trainer import Trainer
from lib.data_loaders import get_dataset

def main(config):

    train_loader = get_dataset(config, "train", config.batch_size, config.max_epoch, shuffle=20)
    test_loader = get_dataset(config, "test", config.test_batch_size, 1)

    trainer = Trainer(config, train_loader, test_loader)

    trainer.train()


if __name__=="__main__":

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    config = get_config()
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    print(tf.test.is_gpu_available())
    main(config)