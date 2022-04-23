import torch
import torch.nn as nn
from torch.nn import functional as f
from models.func.utils import myvif


class Unfold(nn.Module):
    """
    调节stride的大小可以实现overlapping的功能
    input size: B, 1, 1, D (2D 卷积， 需要四维input)
    output size: B, N, D
    permute and unsqueeze: B, N, 1, D
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # a = x.permute(0, 2, 1, 3)
        b = f.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        b = b.permute(0, 2, 1)
        b = b.unsqueeze(-2)
        return b


class ResLayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ResLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBasicBlock(nn.Module):
    "Residual Squeeze-and-Excitation(SE) Block"
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.reslayer = ResLayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.reslayer(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Node2Vec(nn.Module):
    def __init__(self, afr_reduced_cnn_size, dropout):
        super(Node2Vec, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, bias=False, padding=int(49//2)),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=7, stride=4, padding=int(7//2)),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=int(7//2)),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=int(7//2)),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=3, stride=4, padding=int(3//2)),
            nn.Dropout(dropout)
        )

        self.inplanes = 128
        self.AFR = self._make_layer(ResBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.AFR(x)
        return x


class GNA(nn.Module):
    def __init__(self, channel, reduction=2):
        super(GNA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class NodeGenerator(nn.Module):
    def __init__(self, segment_size, overlapping_rate, in_channels, dropout=0.5):
        super(NodeGenerator, self).__init__()
        """
        1. 输入： 需要是一个4D的数据
        2. in_channels是选定segment大小，使用滑动窗口+overlapping后的segments数量，需要计算
        3. 输出： Batch, segments, multi-head, dim
        """
        self.overlapping = int(segment_size * overlapping_rate)
        "滑动窗口，截取segments"
        self.unflod = Unfold((1, segment_size), self.overlapping)
        self.node2vec = Node2Vec(30, dropout)
        self.gna = nn.Sequential(
            GNA(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        node_meta = []
        x = self.unflod(x)
        x = x.squeeze()
        node_size = x.size()[1]
        for idx in range(node_size):
            data = x[:, idx, :]
            data = data.unsqueeze(1)
            out = self.node2vec(data)
            out = out.view(x.size()[0], 1, -1)
            node_meta.append(out)
        node = torch.cat(node_meta, dim=1)
        node = node.unsqueeze(2)
        vif = myvif(node)
        vif = vif.squeeze()
        node = self.gna(node)
        return vif, node


class MultiHead(nn.Module):
    def __init__(self, out_channels):
        super(MultiHead, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        "multi-head"
        x = self.layer(x)
        x = x.permute(0, 2, 1, 3)
        return x


# if __name__ == "__main__":
#     """
#     class PrepBlock(nn.Module) 测试
#     """
#     raw_data = torch.randn(128, 1, 1, 3000)
#     model_1 = NodeGenerator(600, 0.5, 9)
#     model_2 = MultiHead(32)
#     vif, pred_data = model_1(raw_data)
#     print("pred data size:", pred_data.size())
#     print(vif.size())
#     out = model_2(pred_data)
#     print(out.size())