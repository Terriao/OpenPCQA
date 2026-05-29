import mindspore.nn as nn

class Regression(nn.Cell):
    def __init__(self, input):
        super(Regression, self).__init__()
        self.regression = nn.SequentialCell([nn.Dense(input,128),
                                        nn.Dense(128,1),])

    def construct(self, s_img1):
        out = self.regression(s_img1)
        return out
