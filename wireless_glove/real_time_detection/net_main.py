from resnet18 import ResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F


class MainNet(nn.Module):
    def __init__(self, channel_size, feature_size, input_dim, class_num):
        super(MainNet, self).__init__()
        self.resnet = ResNet18()
        self.mlp = nn.Sequential(
            nn.Linear(channel_size * feature_size, channel_size * feature_size),
            nn.GELU(),
            nn.Linear(channel_size * feature_size, 1024)
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, class_num)
        )

    def forward(self, x):
        x = x.view(x.size()[0], 1, 1, -1)
        x = F.normalize(x, p=2, dim=-1)
        x = self.mlp(x)
        x = x.view(x.size()[0], 1, 32, 32)
        x = self.resnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    net = MainNet(8, 250, 512, 3)
    data = torch.randn(20, 1,  8, 250)
    out = net(data)
    print(out.size())



