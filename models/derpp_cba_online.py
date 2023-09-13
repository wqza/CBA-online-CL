# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.ResNet_meta import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Penalty weight.')
    return parser


class DERppCBAonline(ContinualModel):
    NAME = 'derpp-cba-online'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DERppCBAonline, self).__init__(backbone, loss, args, transform)

        self.opt = SGD(self.net.params(), lr=args.lr)
        if args.dataset == 'seq-cifar10' or args.dataset == 'seq-cifar10-blurry':
            meta_lr = 1e-3
        elif args.dataset == 'seq-cifar100' or args.dataset == 'seq-cifar100-blurry':
            meta_lr = 1e-2
        elif args.dataset == 'seq-tinyimg' or args.dataset == 'seq-tinyimg-blurry':
            meta_lr = 1e-2
        self.CBA = MetaCBA(self.num_cls, self.num_cls, hid_dim=256).to(self.device)
        self.opt_cba = Adam(self.CBA.params(), lr=meta_lr)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.current_task = 0
        self.ii = 0

    def observe(self, inputs, labels, not_aug_inputs):
        iter_num = 1
        for ii in range(iter_num):
            if not self.buffer.is_empty():
                buf_inputs1, _, buf_logits1, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                buf_inputs2, buf_labels2, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            # Outer-loop Optimization
            if self.current_task > 0 and self.ii % 5 == 0:
                self.cba_updating(inputs, labels,
                                  buf_inputs1, buf_logits1,
                                  buf_inputs2, buf_labels2)

            # Inner-loop Optimization
            _outputs = self.net(inputs)
            with torch.no_grad():
                res_outputs = self.CBA(F.softmax(_outputs, dim=-1))
            outputs = _outputs + res_outputs
            loss = self.loss(outputs, labels.long())

            if not self.buffer.is_empty():
                _buf_outputs1 = self.net(buf_inputs1)
                _buf_outputs2 = self.net(buf_inputs2)
                with torch.no_grad():
                    res_buf_outputs1 = self.CBA(F.softmax(_buf_outputs1, dim=-1))
                    res_buf_outputs2 = self.CBA(F.softmax(_buf_outputs2, dim=-1))
                buf_outputs1 = _buf_outputs1 + res_buf_outputs1
                buf_outputs2 = _buf_outputs2 + res_buf_outputs2

                loss += self.args.alpha * F.mse_loss(buf_outputs1, buf_logits1)
                loss += self.args.beta * self.loss(buf_outputs2, buf_labels2.long())

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data,  # w/ BA
                                 logits_meta=_outputs.data)  # w/o BA

        return loss.item()

    def cba_updating(self, inputs, labels, buf_inputs1, buf_logits1, buf_inputs2, buf_labels2):
        # 1. copy the model to meta model
        meta_model = self.load_meta_model()
        meta_model.load_state_dict(self.net.state_dict())

        # 2. one step updating virtually
        _outputs = meta_model(inputs)
        _buf_outputs1 = meta_model(buf_inputs1)
        _buf_outputs2 = meta_model(buf_inputs2)

        res_outputs = self.CBA(F.softmax(_outputs.detach(), dim=-1))
        res_buf_outputs1 = self.CBA(F.softmax(_buf_outputs1.detach(), dim=-1))
        res_buf_outputs2 = self.CBA(F.softmax(_buf_outputs2.detach(), dim=-1))

        outputs = _outputs + res_outputs
        buf_outputs1 = _buf_outputs1 + res_buf_outputs1
        buf_outputs2 = _buf_outputs2 + res_buf_outputs2

        loss = self.loss(outputs, labels.long())
        loss += self.args.alpha * F.mse_loss(buf_outputs1, buf_logits1)
        loss += self.args.beta * self.loss(buf_outputs2, buf_labels2.long())

        meta_model.zero_grad()
        grads = torch.autograd.grad(loss, meta_model.fc.params(), create_graph=True)
        meta_model.fc.update_params(lr_inner=self.opt.param_groups[0]['lr'], source_params=grads)
        del grads

        # 3. update bias corrector by buffer set
        buf_inputs, _, _, buf_logits_meta = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
        _buf_outputs = meta_model(buf_inputs)
        loss_outer = F.mse_loss(_buf_outputs, buf_logits_meta)

        buf_inputs, buf_labels, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
        _buf_outputs = meta_model(buf_inputs)
        loss_outer += self.loss(_buf_outputs, buf_labels.long())

        self.opt_cba.zero_grad()
        loss_outer.backward()
        self.opt_cba.step()

    def load_meta_model(self):
        if self.args.backbone == 'resnet18-meta':
            backbone = resnet18_meta(self.num_cls).to(self.device)
        elif self.args.backbone == 'resnet34-meta':
            backbone = resnet34_meta(self.num_cls).to(self.device)
        elif self.args.backbone == 'resnet50-meta':
            backbone = resnet50_meta(self.num_cls).to(self.device)
        elif self.args.backbone == 'resnet101-meta':
            backbone = resnet101_meta(self.num_cls).to(self.device)
        elif self.args.backbone == 'resnet152-meta':
            backbone = resnet152_meta(self.num_cls).to(self.device)
        return backbone
