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
from backbone.meta_adam import MetaAdam


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Penalty weight.')
    return parser


class DERppCBAoffline(ContinualModel):
    # debug: save the logits does not through the BA
    # with BN trick
    NAME = 'derpp-cba-offline'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DERppCBAoffline, self).__init__(backbone, loss, args, transform)

        self.opt = SGD(self.net.params(), lr=args.lr)
        if args.dataset == 'seq-cifar10' or args.dataset == 'seq-cifar10-blurry':
            self.total_classes = 10
            meta_lr = 1e-3
        elif args.dataset == 'seq-cifar100' or args.dataset == 'seq-cifar100-blurry':
            self.total_classes = 100
            meta_lr = 1e-4
        elif args.dataset == 'seq-tinyimg' or args.dataset == 'seq-tinyimg-blurry':
            self.total_classes = 200
            meta_lr = 1e-4
        self.BiasCorrector = MetaBiasCorrector(self.total_classes, self.total_classes, hid_dim=256).to(self.device)
        # self.opt_bc = Adam(self.BiasCorrector.params(), lr=0.01, weight_decay=1e-4)
        self.opt_bc = Adam(self.BiasCorrector.params(), lr=meta_lr)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.current_task = 0
        self.seen_classes = 0
        self.hidden_dim = 512

        self.ii = 0

        self.epoch = 0

    def observe(self, inputs, labels, not_aug_inputs):
        iter_num = 1
        for ii in range(iter_num):
            # update the bias corrector by meta learning
            if not self.buffer.is_empty():
                buf_inputs1, _, buf_logits1, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                buf_inputs2, buf_labels2, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            else:
                buf_inputs1, buf_logits1 = None, None
                buf_inputs2, buf_labels2 = None, None

            if self.current_task > 0 and self.ii % 5 == 0:
            # if self.current_task > 0 and self.ii % (5 * self.args.n_epochs) == 0:
                self.bias_corrector_updating(inputs, labels, buf_inputs1, buf_logits1, buf_inputs2, buf_labels2)

            # update the total network by new task and buffer samples
            self.net.apply(bn_no_momentum)
            _outputs = self.net(inputs)
            with torch.no_grad():
                res_outputs = self.BiasCorrector(F.softmax(_outputs, dim=-1))
            outputs = _outputs + res_outputs
            loss = self.loss(outputs, labels.long())

            if not self.buffer.is_empty():
                self.net.apply(bn_normal_momentum)
                _buf_outputs1 = self.net(buf_inputs1)
                _buf_outputs2 = self.net(buf_inputs2)
                with torch.no_grad():
                    res_buf_outputs1 = self.BiasCorrector(F.softmax(_buf_outputs1, dim=-1))
                    res_buf_outputs2 = self.BiasCorrector(F.softmax(_buf_outputs2, dim=-1))
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

    def bias_corrector_updating(self, inputs, labels, buf_inputs1, buf_logits1, buf_inputs2, buf_labels2):
        # update the bias corrector by meta learning
        # 1. copy the model to meta model
        if self.args.backbone == 'resnet18-meta':
            meta_model = resnet18_meta(self.total_classes).to(self.device)
        meta_model.load_state_dict(self.net.state_dict())

        # 2. one step updating virtually
        _outputs = meta_model(inputs)
        _buf_outputs1 = meta_model(buf_inputs1)
        _buf_outputs2 = meta_model(buf_inputs2)

        res_outputs = self.BiasCorrector(F.softmax(_outputs.detach(), dim=-1))
        res_buf_outputs1 = self.BiasCorrector(F.softmax(_buf_outputs1.detach(), dim=-1))
        res_buf_outputs2 = self.BiasCorrector(F.softmax(_buf_outputs2.detach(), dim=-1))

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

        # 3. update bias corrector by meta set
        # (here we use samples from the memory buffer as the balanced meta set temporally)
        buf_inputs, _, _, buf_logits_meta = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
        _buf_outputs = meta_model(buf_inputs)
        loss_meta = F.mse_loss(_buf_outputs, buf_logits_meta)

        buf_inputs, buf_labels, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
        _buf_outputs = meta_model(buf_inputs)
        loss_meta += self.loss(_buf_outputs, buf_labels.long())

        self.opt_bc.zero_grad()
        loss_meta.backward()
        self.opt_bc.step()


def bn_no_momentum(m):
    if isinstance(m, MetaBatchNorm2d):
        m.momentum = 0.0


def bn_normal_momentum(m):
    if isinstance(m, MetaBatchNorm2d):
        m.momentum = 0.1




