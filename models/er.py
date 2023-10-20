# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch.nn.functional as F

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.epoch = 0
        self.task = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        # self.opt.zero_grad()
        # if not self.buffer.is_empty():
        #     buf_inputs, buf_labels = self.buffer.get_data(
        #         self.args.minibatch_size, transform=self.transform)
        #     inputs = torch.cat((inputs, buf_inputs))
        #     labels = torch.cat((labels, buf_labels))
        #
        # outputs = self.net(inputs)
        # loss = self.loss(outputs, labels.long())
        # loss.backward()
        # self.opt.step()

        # split er
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size,
                                                          transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.loss(buf_outputs, buf_labels.long())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

