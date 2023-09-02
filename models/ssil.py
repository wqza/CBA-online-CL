# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import numpy as np
import torch
import copy
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.ResNet import resnet18


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class SSIL(ContinualModel):
    NAME = 'ssil'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SSIL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.current_task = 0  # start counting from 0
        self.n_cls_per_task = 2
        self.total_classes = 10
        self.T = 2  # temperature
        self.old_net = None
        # self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

        self.pause = False
        self.seen_classes = [0]

    def observe(self, inputs, labels, not_aug_inputs):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        curr_size = inputs.shape[0]
        mid = self.current_task * self.n_cls_per_task
        end = (self.current_task + 1) * self.n_cls_per_task
        if self.current_task == 0:  # the first task
            outputs = self.net(inputs)
            loss = self.loss(outputs[:curr_size, mid:end], labels)

        else:
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            prev_size = buf_inputs.shape[0]

            cat_inputs = torch.cat((inputs, buf_inputs))
            outputs = self.net(cat_inputs)

            # separated softmax
            labels = labels % (end - mid)
            loss_CE_curr = self.loss(outputs[:curr_size, mid:end], labels.long(), reduction='sum')
            loss_CE_prev = self.loss(outputs[curr_size:, :mid], buf_labels.long(), reduction='sum')
            loss_CE = (loss_CE_curr + loss_CE_prev) / (curr_size + prev_size)
            # import pdb; pdb.set_trace()

            # task-wise KD
            score = self.old_net(cat_inputs)[:, :mid].data  # the old model trained on the last task
            loss_KD = torch.zeros(self.current_task).to(self.device)
            for tt in range(self.current_task):
                start_KD = self.n_cls_per_task * tt
                end_KD = self.n_cls_per_task * (tt + 1)

                soft_target = F.softmax(score[:, start_KD:end_KD] / self.T, dim=1)
                output_log = F.log_softmax(outputs[:, start_KD:end_KD] / self.T, dim=1)
                loss_KD[tt] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.T ** 2)

            loss_KD = loss_KD.sum()
            loss = loss_CE + loss_KD

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    # def end_task(self, train_loader):
    def end_task(self, dataset):
        # copy the model to the old model
        self.old_net = resnet18(self.total_classes).to(self.device)
        self.old_net.load_state_dict(self.net.state_dict())
        # self.old_net = copy.deepcopy(self.net.eval())

        self.fill_buffer(self.buffer, dataset, self.current_task)

    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)
        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y = self.buffer.get_all_data()

            mem_buffer.empty()
            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y = buf_x[idx], buf_y[idx]
                mem_buffer.add_data(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class]
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        a_x, a_y = [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
        a_x, a_y = torch.cat(a_x), torch.cat(a_y)

        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y = a_x[idx], a_y[idx]

            randind = torch.randperm(_x.size(0))[:samples_per_class]

            mem_buffer.add_data(
                examples=_x[randind].to(self.device),
                labels=_y[randind].to(self.device)
            )
        assert len(mem_buffer.examples) <= mem_buffer.buffer_size
