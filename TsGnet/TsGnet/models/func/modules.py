
import torch
from torch.nn import functional as f
import torch.nn as nn
from models.func.utils import mycorr


class Attention(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.nolinear = nn.Sigmoid()

    def forward(self, x):
        B, M, N, _ = x.size()
        adj = mycorr(x)
        x_ = self._prepare_attentional_mechanism_input(x)
        e = self.avg_pool(x_)
        e = e.permute(0, 2, 1, 3)
        e = self.fc(e)
        e = e.permute(0, 2, 1, 3)
        e = e.view(B, M, N, N)
        e = self.nolinear(e)
        zero_vec_adj = -1e12 * torch.ones_like(adj)
        attention = torch.where(adj > 0, e, zero_vec_adj)
        attention = f.softmax(attention, dim=-1)
        x = torch.matmul(attention, x)
        return x

    def _prepare_attentional_mechanism_input(self, Wh):
        B, M, N, D = Wh.size()  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=-2)  # 对每一个特征向量重复N次
        Wh_repeated_alternating = Wh.repeat(1, 1, N, 1)  # 将特征矩阵重复N次
        pos = 2 * torch.ones_like(Wh_repeated_alternating)
        one_vec = torch.ones_like(Wh_repeated_alternating)
        pos = torch.where(Wh_repeated_in_chunks - Wh_repeated_alternating == 0, one_vec, pos)
        Wh_repeated_alternating = pos * Wh_repeated_alternating
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        # all_combinations_matrix = all_combinations_matrix.permute(0, 2, 1, 3)
        return all_combinations_matrix


class Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride):
        super().__init__()
        """
        in_channels: segments
        mid_channels: last multi-head size
        out_channels: new multi-head size
        """
        self.attn = Attention(in_channels*in_channels)
        padding = kernel_size//2
        self.mid_channels_ = int((mid_channels - kernel_size + 2 * padding) // stride + 1)

        self.layer_1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            self.attn,
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)

        )

        self.layer_3 = nn.Conv2d(self.mid_channels_, out_channels, 1, bias=False)

        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride = stride

    def forward(self, x):
        """
        :input x: Batch, node, multi-head, dim
        :return: Batch, multi-head, node, dim
        """

        """
        shortcut
        """
        res = x.permute(0, 2, 1, 3)
        shortcut = self.shortcut(res)
        if self.stride != 1:
            shortcut = torch.sum(shortcut, dim=-2).unsqueeze(-2)

        """
        Globel Node Attention
        """
        x = x.permute(0, 2, 1, 3)
        out = self.layer_1(x)
        out = out.permute(0, 2, 1, 3)

        """
        extract features from every node channels by using group
        Batch, node, multi-head, dim
        """
        out = self.layer_2(out)
        """
        Batch, multi-head, node, dim
        """
        out = out.permute(0, 2, 1, 3)
        """
        increase the heads
        # """
        out = self.layer_3(out)
        out += shortcut
        out = out.permute(0, 2, 1, 3)
        return out


# if __name__ == "__main__":
#     """
#     class Block(nn.Module) 测试
#     """
#     data = torch.randn(128, 9, 32, 200)
#     model_2 = Block(9, 32, 64, 7, 2)
#     out_1 = model_2(data)
#     print("out_1 size", out_1.size())

