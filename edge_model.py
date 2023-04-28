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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Model at edge side
class ResNet18_edge_side(nn.Module):
    def __init__(self, block, num_layers, classes, inplanes=64, planes=64, stride=1, downsample=None):
        super(ResNet18_edge_side, self).__init__()
        self.inplanes = 64
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
        self.layer3 = self._make_layer(block, 64, num_layers[0], stride=2)
        self.layer4 = self._make_layer(block, 128, num_layers[1], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Model at edge side auxiliary network
class ResNet18_edge_side_auxiliary(nn.Module):
    def __init__(self):
        super(ResNet18_edge_side_auxiliary, self).__init__()
        self.averagePool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 14)

    def forward(self, x):
        x = self.averagePool(x)
        x = x.view(-1, 128)
        x = self.fc(x)

        return x


net_glob_edge = ResNet18_edge_side(block=BasicBlock, num_layers=[1, 1], classes=14)
net_glob_edge_auxiliary = ResNet18_edge_side_auxiliary()
net_glob_edge2 = ResNet18_edge_side(block=BasicBlock, num_layers=[1, 1], classes=14)
net_glob_edge2_auxiliary = ResNet18_edge_side_auxiliary()
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
edge_side_model = ResNet18_edge_side(block=BasicBlock, num_layers=[1, 1], classes=14).to(device)
edge_side_auxiliary_model = ResNet18_edge_side_auxiliary().to(device)
# 传入model和输入数据的shape
summary(edge_side_model, (64,17))
summary(edge_side_auxiliary_model, (64, 17))
