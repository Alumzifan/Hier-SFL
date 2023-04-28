import torch
from torch import nn
import math
import torch.nn.functional as F
from setup import device
from torchsummary import summary

# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self):
        super(ResNet18_client_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2 = out2 + out1  # adding the resudial inputs -- downsampling not required in this layer
        resudial = F.relu(out2)
        return resudial


# Model at client side auxiliary network
class ResNet18_client_side_auxiliary(nn.Module):
    def __init__(self):
        super(ResNet18_client_side_auxiliary, self).__init__()
        self.averagePool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 14)

    def forward(self, x):
        x = self.averagePool(x)
        x = x.view(-1, 64)
        x = self.fc(x)

        return x


net_glob_client = ResNet18_client_side()
net_glob_client_auxiliary = ResNet18_client_side_auxiliary()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)
    net_glob_client_auxiliary = nn.DataParallel(net_glob_client_auxiliary)

net_glob_client.to(device)
net_glob_client_auxiliary.to(device)
print(net_glob_client)
print(net_glob_client_auxiliary)

# 打印客户端网络模型
# Discriminator()是我需要可视化的判别器
client_side_model = ResNet18_client_side().to(device)
client_side_auxiliary_model = ResNet18_client_side_auxiliary().to(device)
# 传入model和输入数据的shape
summary(client_side_model, (1, 66))
summary(client_side_auxiliary_model, (64, 17))
