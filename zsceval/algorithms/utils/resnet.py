import math

import torch.nn as nn
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, input_channles, layers, num_classes=1000):
        self.in_planes = 64
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channles, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # [batch_size * seq_len, 1, input_size, input_size]
        x = self.conv1(x)
        # [batch_size * seq_len, 64, input_size // 2, input_size // 2]
        x = self.max_pool(x)
        # [batch_size * seq_len, 64, input_size // 4, input_size // 4]
        x = self.layer1(x)
        # [batch_size * seq_len, 64, input_size // 4, input_size // 4]
        x = self.layer2(x)
        # [batch_size * seq_len, 64, input_size // 8, input_size // 8]
        x = self.layer3(x)
        # [batch_size * seq_len, 64, input_size // 16, input_size // 16]
        x = self.layer4(x)
        # [batch_size * seq_len, 64, input_size // 32, input_size // 32]
        x = self.global_avg_pool(x)
        # [batch_size * seq_len, 512 * expansion, 1, 1]
        x = x.view(x.size(0), -1)
        # [batch_size * seq_len, 512 * expansion]
        x = self.fc(x)
        # [batch_size * seq_len, num_classes]
        return x


class MapNet(nn.Module):
    def __init__(self, inputs_channels, num_classes, layers):
        super().__init__()

        self.resnet = ResNet(BasicBlock, inputs_channels, layers, num_classes)

    def forward(self, x):
        # [batch_size, 1, input_size, input_size]
        x = self.resnet(x)
        # [batch_size, num_classes]
        return x


class Pre_MapNet(nn.Module):
    def __init__(self, inputs_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inputs_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        resnet = models.resnet18(pretrained=1)
        modules = list(resnet.children())[:-1]  # 去除最后的fc层
        self.resnet_base = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # [batch_size, 1, input_size, input_size]
        x = self.conv1(x)
        # [batch_size, num_classes]
        x = self.resnet_base(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
