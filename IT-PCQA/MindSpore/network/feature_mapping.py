import mindspore.nn as nn

class Feature_mapping(nn.Cell):
  def __init__(self, in_feature, hidden_size):
    super(Feature_mapping, self).__init__()
    self.ad_layer1 = nn.Dense(in_feature, hidden_size)
    self.ad_layer2 = nn.Dense(hidden_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()

  def construct(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    return x
