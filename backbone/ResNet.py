"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
        Basic Block for resnet 18 and resnet 34
        BasicBlock and BottleNeck block have different output size we use class attribute expansion to distinct
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
        Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)

        # self._features = nn.Sequential(self.conv1,
        #                                self.conv2_x,
        #                                self.conv3_x,
        #                                self.conv4_x,
        #                                self.conv5_x
        #                                )
        # self.classifier = self.fc

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

    def forward(self, x, returnt='out', norm=False, noise=None, feature=None):
        if feature is not None:
            output = self.fc(feature)
            return output
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        feature = output.view(output.size(0), -1)

        if returnt == 'feature-aug':
            if noise is None:
                a = torch.normal(1, 1, size=feature.shape).to(feature.device)
                b = torch.normal(0, 1, size=feature.shape).to(feature.device)
                feature = feature * a + b
            else:
                feature = feature + noise
            feature = F.normalize(feature, p=2, dim=1, eps=1e-12)
            output = self.fc(feature)
            return output

        if norm:
            # feature = F.normalize(feature, p=2, dim=1)
            linear_weight = F.normalize(self.fc.weight, p=2, dim=1)
            output = F.linear(feature, linear_weight)
            # output = self.fc(feature)

            # # scale_factor = 10
            # beta = 0.5
            #
            # # normalize feature by power transform
            # # feature_norm = torch.norm((feature + 1e-6) ** beta, 2, dim=1).unsqueeze(1).expand_as(feature)
            # # feature = (feature + 1e-6) ** beta
            # # feature = feature.div(feature_norm)
            # #
            # # # normalize fc layer by dividing the l2-norm
            # # fc_norm = torch.norm(self.fc.weight.data, 2, dim=1).unsqueeze(1).expand_as(self.fc.weight.data)
            # # self.fc.weight.data = self.fc.weight.data.div(fc_norm)
            # #
            # # output = scale_factor * self.fc(feature)
            #
            # feature = torch.pow(feature, beta)
            # output = self.fc(feature)
        else:
            output = self.fc(feature)

        if returnt == 'out':
            return output
        elif returnt == 'all':
            return output, feature
        elif returnt == 'feature':
            return feature


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)

