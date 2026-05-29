import mindspore as ms
import numpy as np
import time

from config import get_config
from lib.trainer import Trainer
from lib.data_loaders import get_dataloader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config):

    train_loader = get_dataloader(config, "train")
    test_loader = get_dataloader(config, "test")

    trainer = Trainer(config, train_loader, test_loader)
    
    if config.phase=='train':
        trainer.train()
    if config.phase=='test':
        trainer._test_epoch()


if __name__=="__main__":
    config = get_config()
    main(config)