import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.func.modules import Block
from models.func.node_generator import NodeGenerator, MultiHead


class PearsonGAT(nn.Module):
    def __init__(self, segment_size, overlapping_rate, in_channels, out_channels):
        super(PearsonGAT, self).__init__()
        self.segment_size = segment_size
        self.node_generation = NodeGenerator(segment_size, overlapping_rate, in_channels)
        self.multi_head = MultiHead(out_channels)

        self.attn = nn.Sequential(
            Block(in_channels, 16, 16, 5, 1),
            Block(in_channels, 16, 32, 5, 2),
            Block(in_channels, 32, 32, 5, 1),
            Block(in_channels, 32, 64, 5, 2),
            Block(in_channels, 64, 64, 5, 1),
            Block(in_channels, 64, 128, 5, 2),
            Block(in_channels, 128, 128, 5, 1),
            Block(in_channels, 128, 128, 5, 2),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(9*128*14, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Conv2d(256, 5, 1, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        vif, x = self.node_generation(x)
        x = self.multi_head(x)
        x = self.attn(x)
        x = x.flatten(1).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        return vif, F.log_softmax(x, dim=-1)
        # return vif, x



# if __name__=="__main__":
#     a = torch.randn(256, 1, 3000)
#     print()
#     print("raw data size:")
#     print(a.size())
#     print()
#     print("+++++++++++++++++++++++++++++++++++++")
#     model = PearsonGAT(segment_size=600, overlapping_rate=0.5, in_channels=9, out_channels=16)
#     # vif, out = model(a)
#     # print("model out size:")
#     # print(out.size())
#     # print(vif.size())
#
#     _, out = model(a)
#     print("model out size:")
#     print(out.size())






