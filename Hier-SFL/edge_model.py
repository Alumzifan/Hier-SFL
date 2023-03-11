import math
import torch
from torch import nn
import torch.nn.functional as F
from setup import device
from torchsummary import summary


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Model at edge side
class Lightweight_ResNet18_edge_side(nn.Module):
    def __init__(self, inplanes=64, planes=64, stride=1, downsample=None):
        super(Lightweight_ResNet18_edge_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
        )
        self.layer2 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        resudial1 = self.layer1(x)
        resudial1 = self.layer2(resudial1)
        return resudial1


# Model at edge side auxiliary network
class Lightweight_ResNet18_edge_side_auxiliary(nn.Module):
    def __init__(self):
        super(Lightweight_ResNet18_edge_side_auxiliary, self).__init__()
        self.averagePool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 8)

    def forward(self, x):
        x = self.averagePool(x)
        x = x.view(-1, 64)
        x = self.fc(x)

        return x


net_glob_edge = Lightweight_ResNet18_edge_side()
net_glob_edge_auxiliary = Lightweight_ResNet18_edge_side_auxiliary()
net_glob_edge2 = Lightweight_ResNet18_edge_side()
net_glob_edge2_auxiliary = Lightweight_ResNet18_edge_side_auxiliary()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_edge = nn.DataParallel(net_glob_edge)
    net_glob_edge_auxiliary = nn.DataParallel(net_glob_edge_auxiliary)

net_glob_edge.to(device)
net_glob_edge_auxiliary.to(device)
print(net_glob_edge)
print(net_glob_edge_auxiliary)

# 打印客户端网络模型
# Discriminator()是我需要可视化的判别器
edge_side_model = Lightweight_ResNet18_edge_side().to(device)
edge_side_auxiliary_model = Lightweight_ResNet18_edge_side_auxiliary().to(device)
# 传入model和输入数据的shape
summary(edge_side_model, (64,8))
summary(edge_side_auxiliary_model, (64, 8))