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
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        out = self.relu(out)

        return out

class Lightweight_ResNet18_cloud_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(Lightweight_ResNet18_cloud_side, self).__init__()
        self.inplanes = 64
        # self.layer3 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        # )
        self.layer2 = self._make_layer(block, 64, num_layers[0], stride=2)
        self.layer3 = self._make_layer(block, 128, num_layers[1], stride=2)
        # self.layer4 = self._make_layer(block, 256, num_layers[2], stride=2)
        # self.layer6 = self._make_layer(block, 256, num_layers[3], stride=2)
        self.averagePool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def _layer(self, block, planes, num_layers, stride=2):
    #     dim_change = None
    #     if stride != 1 or planes != self.input_planes * block.expansion:
    #         dim_change = nn.Sequential(
    #             nn.Conv1d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
    #             nn.BatchNorm1d(planes * block.expansion))
    #     netLayers = []
    #     netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
    #     self.input_planes = planes * block.expansion
    #     for i in range(1, num_layers):
    #         netLayers.append(block(self.input_planes, planes))
    #         self.input_planes = planes * block.expansion
    #
    #     return nn.Sequential(*netLayers)

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
        # out2 = self.layer3(x)
        # out2 = out2 + x  # adding the resudial inputs -- downsampling not required in this layer
        # x3 = F.relu(out2)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x6 = self.layer6(x5)

        x = self.averagePool(x)
        x = x.view(-1, 128 * 1)
        y_hat = self.fc(x)

        return y_hat


net_glob_cloud = Lightweight_ResNet18_cloud_side(BasicBlock, [1, 1], 8)  # 8 is my numbr of classes
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_cloud = nn.DataParallel(net_glob_cloud)  # to use the multiple GPUs

net_glob_cloud.to(device)
print(net_glob_cloud)

# 打印服务器端端网络模型
# Discriminator()是我需要可视化的判别器
cloud_side_model = Lightweight_ResNet18_cloud_side(BasicBlock, [1, 1], 8).to(device)
# 传入model和输入数据的shape
summary(cloud_side_model, (64, 8))