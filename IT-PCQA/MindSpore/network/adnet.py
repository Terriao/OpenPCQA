import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class AdversarialNetwork(nn.Cell):
  def __init__(self, hidden_size):
    super(AdversarialNetwork, self).__init__()

    self.ad_layer3 = nn.SequentialCell([nn.Dense(hidden_size, 64),
                                   nn.Dense(64, 1),
                                   ])
    self.sigmoid = nn.Sigmoid()

  def construct(self, xfeature, yout, D_s, D_t, source_size, target_size): #D_t=0
    y = self.ad_layer3(xfeature)
    y = self.sigmoid(y)
    dc_target = Tensor(np.array([[1]] * source_size + [[0]] * target_size), ms.float32)
    Dfake = Tensor(np.array([[D_s]] * source_size + [[D_t]] * target_size), ms.float32)
    y = ops.abs(y - Dfake)
    return nn.BCELoss(reduction='mean')(y, dc_target)
