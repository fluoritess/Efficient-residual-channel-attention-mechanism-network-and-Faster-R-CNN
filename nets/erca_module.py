import torch
from torch import nn
from torch.nn.parameter import Parameter
import math

# #ERCA
class ERCALayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(ERCALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual_abs = torch.abs(x)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        avg = self.avg_pool(residual_abs)
        r = y.mul(avg)

        sub = residual_abs.sub(r)
        zeros = sub.sub(sub)
        n_sub_ = torch.gt(sub, zeros)
        n_sub_ = n_sub_.int()
        n_sub = sub.mul(n_sub_)
        out = torch.sign(x).mul(n_sub)

        return out + x

# #ERCAMA
class ERCAMA_Layer(nn.Module):
    def __init__(self, channels, gamma=2, b=1,pool_types=['avg', 'max']):
        super(ERCAMA_Layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool_types = pool_types

    def forward(self, x):
        residual_abs = torch.abs(x)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw=self.conv(avg_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            elif pool_type=='max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.conv(max_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        y = self.sigmoid(channel_att_sum)

        avg = self.avg_pool(residual_abs)
        r = y.mul(avg)

        sub = residual_abs.sub(r)
        zeros = sub.sub(sub)
        n_sub_ = torch.gt(sub, zeros)
        n_sub_ = n_sub_.int()
        n_sub = sub.mul(n_sub_)
        out = torch.sign(x).mul(n_sub)

        return out + x

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        #return x * y.expand_as(x)
        return x * y.expand_as(x)

class ECALayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)

