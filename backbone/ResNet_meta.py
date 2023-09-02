"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .meta_layers import MetaModule, MetaLinear, MetaConv2d, MetaBatchNorm2d, MetaBatchNorm1d


class MetaBasicBlock(MetaModule):
    """
        Basic Block for resnet 18 and resnet 34
        BasicBlock and BottleNeck block have different output size we use class attribute expansion to distinct
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            MetaConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels * MetaBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            MetaBatchNorm2d(out_channels * MetaBasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != MetaBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_channels, out_channels * MetaBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(out_channels * MetaBasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class MetaBottleNeck(MetaModule):
    """
        Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            MetaConv2d(in_channels, out_channels, kernel_size=1, bias=False),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels * MetaBottleNeck.expansion, kernel_size=1, bias=False),
            MetaBatchNorm2d(out_channels * MetaBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * MetaBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_channels, out_channels * MetaBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                MetaBatchNorm2d(out_channels * MetaBottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class MetaResNet(MetaModule):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            MetaConv2d(3, 64, kernel_size=3, padding=1, bias=False),
            MetaBatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MetaLinear(512 * block.expansion, num_classes)
        # self.fc = MetaLinear(512 * block.expansion, num_classes, bias=False)
        # self.fc2 = MetaLinear(512 * block.expansion, num_classes, bias=False)
        # self.fc3 = MetaLinear(512 * block.expansion, num_classes, bias=False)
        # self.fc4 = MetaLinear(512 * block.expansion, num_classes, bias=False)
        # self.fc5 = MetaLinear(512 * block.expansion, num_classes, bias=False)
        # self.fc = MetaClassifier(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the same as a neuron netowork layer, ex. conv layer),
        one layer may contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, returnt='out',
                norm=False, drift_attractor=None, mixup=None, noise_weight=0.0, perturbation=None):

        if returnt == 'fc':
            return self.fc(x)

        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        # output = self.avg_pool(output)
        output = F.avg_pool2d(output, output.shape[2])
        feature = output.view(output.size(0), -1)

        if norm:
            feature = feature / torch.norm(feature, dim=1).unsqueeze(1)

        if drift_attractor is not None:
            with torch.no_grad():
                feature_drift = drift_attractor(feature)
            feature = feature + feature_drift

        if mixup is not None:
            y, x1_feats, y1 = mixup[0].detach(), mixup[1].detach(), mixup[2].detach()
            combined = torch.cat([y.unique(), y1.unique()])
            uniques, counts = combined.unique(return_counts=True)
            intersection_cls = uniques[counts > 1]
            for cc in range(len(intersection_cls)):
                # w = 0.1
                w = 0.2
                # alpha = 0.2
                # w = torch.distributions.beta.Beta(alpha, alpha).sample()
                # if w > 0.5: w = 1 - w
                cls = int(intersection_cls[cc])
                idx = np.random.choice(np.arange(np.array(sum(y1 == cls).cpu())), np.array(sum(y == cls).cpu()))
                feature[y == cls] = feature[y == cls] * (1 - w) + x1_feats[y1 == cls][idx] * w

        if noise_weight > 0:
            gaussian_noise = torch.normal(mean=torch.zeros_like(feature), std=0.1).to(feature.device)
            feature = feature + gaussian_noise * noise_weight

        if perturbation is not None:
            feature += perturbation

        if returnt == 'out':
            output = self.fc(feature)
            return output
        elif returnt == 'all':
            output = self.fc(feature)
            return output, feature
        elif returnt == 'feature':
            return feature


class MetaClassifier(MetaModule):
    def __init__(self, in_dim, out_dim):
        super(MetaClassifier, self).__init__()
        self.fc = MetaLinear(in_dim, out_dim)

        for m in self.modules():
            if isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


class MetaCBA(MetaModule):
    def __init__(self, in_dim, out_dim, hid_dim=512):
        super(MetaCBA, self).__init__()
        self.fc = nn.Sequential(
            MetaLinear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            MetaLinear(hid_dim, out_dim)
        )

        for m in self.modules():
            if isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, MetaBatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def resnet18_meta(num_classes):
    """ return a ResNet 18 object
    """
    return MetaResNet(MetaBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34_meta(num_classes):
    """ return a ResNet 34 object
    """
    return MetaResNet(MetaBasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50_meta(num_classes):
    """ return a ResNet 50 object
    """
    return MetaResNet(MetaBottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101_meta(num_classes):
    """ return a ResNet 101 object
    """
    return MetaResNet(MetaBottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152_meta(num_classes):
    """ return a ResNet 152 object
    """
    return MetaResNet(MetaBottleNeck, [3, 8, 36, 3], num_classes=num_classes)

