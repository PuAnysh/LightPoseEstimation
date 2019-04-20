import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
from networks.DUC import DUC

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel , k_s = 3):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, k_s, stride, k_s//2, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, k_s, stride, k_s//2, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))
        # print(channel_shuffle(out, 2).shape)
        return channel_shuffle(out, 2)


class ShuffleNet(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNet, self).__init__()
        
        assert input_size % 32 == 0
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                # print(input_channel)
                # print(output_channel)
                stride = 2 if i == 0 and idxstage == 0 else 1
                if i == 0:
                    #inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
                
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last      = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.conv_compress = nn.Conv2d(1024, 256, 1, 1, 0, bias=False)
        self.stage = 6
        self.cpm = []
        self.cpm_channel = self.stage_out_channels[-1]
        for i in range(self.stage):
            kernel_size = 7
            lastest_channel_size = 128
            if i == 0:
                kernel_size = 3
                lastest_channel_size = 512
            cpm_stage = nn.Sequential(
                InvertedResidual(self.cpm_channel, 32, 1, 2,kernel_size),
                InvertedResidual(32, 32, 1, 2, kernel_size),
                InvertedResidual(32, 32, 1, 2, kernel_size),
                nn.Conv2d(32, lastest_channel_size, 1, 1, 0, bias=False),
                nn.Conv2d(lastest_channel_size, n_class, 1, 1, 0, bias=False)
            )
            self.cpm_channel = n_class + self.stage_out_channels[-1]
            self.cpm.append(cpm_stage)
        self.cpm = nn.Sequential(*self.cpm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        feature = self.features(x)
        feature = self.conv_last(feature)
        x = feature
        heatmaps = []
        for i, stage in enumerate(self.cpm):
            x = stage(x)
            heatmaps.append(x)
            x = torch.cat([x, feature], 1)
        return heatmaps


def shufflenetv2_ed(width_mult=1.):
    model = ShuffleNet(width_mult=width_mult)
    model = nn.Sequential(*(list(model.children())[:-3]))
    for param in model.parameters():
        param.requires_grad = True
    return model

if __name__ == '__main__':
    model = ShuffleNet(16)
    print(model)