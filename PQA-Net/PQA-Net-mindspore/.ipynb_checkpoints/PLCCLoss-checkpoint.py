# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
# [GCC 7.5.0]
# Embedded file name: /userhome/BBBBBBBBB/PQA-Net-mindspore/PLCCLoss.py
# Compiled at: 2022-12-14 09:49:04
# Size of source mod 2**32: 1033 bytes
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class PLCCLoss(nn.Cell):

    def __init__(self):
        super(PLCCLoss, self).__init__()

    def construct(self, input, target):
        input0 = input - ops.mean(input)
        target0 = target - ops.mean(target)
        self.loss = ops.sum(input0 * target0) / (ops.sqrt(ops.sum(input0 ** 2)) * ops.sqrt(ops.sum(target0 ** 2)))
        return self.loss