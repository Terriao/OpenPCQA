# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
# [GCC 7.5.0]
# Embedded file name: /userhome/BBBBBBBBB/PQA-Net-mindspore/Gdn.py
# Compiled at: 2023-01-11 16:29:18
# Size of source mod 2**32: 4322 bytes
import mindspore, mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

class GdnFunction(nn.Cell):

    def __init__(self):
        super(GdnFunction, self).__init__()

    def construct(self, x, gamma, beta):
        n, c, h, w = list(x.shape)
        tx = x.permute((0, 2, 3, 1))
        tx = tx.view((-1, c))
        tx2 = tx * tx
        denominator = ms.ops.matmul(tx2, gamma) + beta
        ty = tx / ops.sqrt(denominator)
        y = ty.view(n, h, w, c)
        y = y.permute(0, 3, 1, 2)
        
        return y

    def bprop(self, x, gamma, beta, out, grad_output):
        n, c, h, w = list(grad_output.shape)
        tx = x.permute((0, 2, 3, 1))
        tx = tx.view((-1, c))
        tx2 = tx * tx
        denominator = ms.ops.matmul(tx2, gamma) + beta
        tdzdy = grad_output.permute((0, 2, 3, 1))
        tdzdy = tdzdy.view((-1, c))
        gy = tdzdy * ops.pow(denominator, -0.5) - ops.matmul(tdzdy * tx * ops.pow(denominator, -1.5), gamma.transpose(1, 0)) * tx
        gy = gy.view(n, h, w, c)
        grad_input = gy.permute((0, 3, 1, 2))
        tmp = -0.5 * ops.pow(denominator, -1.5) * tx * tdzdy
        grad_beta = tmp.sum(axis=0)
        grad_gamma = ops.matmul(tx2.transpose(1, 0), tmp)
        return (
         grad_input, grad_gamma, grad_beta)


class Gdn(nn.Cell):

    def __init__(self, input_channel):
        super(Gdn, self).__init__()
        self.input_channel = input_channel
        self.gamma = ms.Parameter((ms.Tensor(np.random.uniform(0, 1, (input_channel, input_channel)), ms.float32)), name='gamma', requires_grad=True)
        self.beta = ms.Parameter((ms.Tensor(np.random.uniform(0, 1, input_channel), ms.float32)), name='beta', requires_grad=True)

    def construct(self, x):
        return GdnFunction()(x, self.gamma, self.beta)

    def __str__(self):
        return self.__class__.__name__ + '(gamma_size=(%d, %d), beta_size=(%d))' % (
         self.gamma.shape[0], self.gamma.shape[1], self.beta.shape[0])

    __repr__ = __str__