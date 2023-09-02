# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.conf import data_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])

    # TRANSFORM = transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                                 transforms.RandAugment(1, 14),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                                      (0.2470, 0.2435, 0.2615))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(data_path() + 'CIFAR10', train=True, download=False, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(data_path() + 'CIFAR10', train=False, download=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(data_path() + 'CIFAR10', train=True, download=False, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose([transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(backbone_name='resnet18'):
        if backbone_name == 'resnet18':
            from backbone.ResNet import resnet18
            backbone = resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
            # from backbone.ResNet18 import resnet18
            # return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet34':
            from backbone.ResNet import resnet34
            backbone = resnet34(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet50':
            from backbone.ResNet import resnet50
            backbone = resnet50(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet101':
            from backbone.ResNet import resnet101
            backbone = resnet101(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet152':
            from backbone.ResNet import resnet152
            backbone = resnet152(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)

        if backbone_name == 'resnet18-meta':
            from backbone.ResNet_meta import resnet18_meta
            backbone = resnet18_meta(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet34-meta':
            from backbone.ResNet_meta import resnet34_meta
            backbone = resnet34_meta(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet50-meta':
            from backbone.ResNet_meta import resnet50_meta
            backbone = resnet50_meta(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet101-meta':
            from backbone.ResNet_meta import resnet101_meta
            backbone = resnet101_meta(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        elif backbone_name == 'resnet152-meta':
            from backbone.ResNet_meta import resnet152_meta
            backbone = resnet152_meta(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None
