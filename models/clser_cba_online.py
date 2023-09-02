import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F

from torch.optim import SGD, Adam
from backbone.ResNet_meta import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class CLSERCBAonline(ContinualModel):
    NAME = 'clser-cba-online'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CLSERCBAonline, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

        self.opt = SGD(self.net.params(), lr=args.lr)
        if args.dataset == 'seq-cifar10' or args.dataset == 'seq-cifar10-blurry':
            meta_lr = 1e-3
        elif args.dataset == 'seq-cifar100' or args.dataset == 'seq-cifar100-blurry':
            meta_lr = 1e-2
        elif args.dataset == 'seq-tinyimg' or args.dataset == 'seq-tinyimg-blurry':
            meta_lr = 1e-2
        self.CBA = MetaCBA(self.num_cls, self.num_cls, hid_dim=256).to(self.device)
        self.opt_cba = Adam(self.CBA.params(), lr=meta_lr)

        self.ii = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        # Outer-loop Optimization
        if self.current_task > 0 and self.ii % 5 == 0:
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            self.cba_updating(inputs, labels, buf_inputs, buf_labels)

        # Inner-loop Optimization
        self.opt.zero_grad()

        _outputs = self.net(inputs)
        with torch.no_grad():
            res_outputs = self.CBA(F.softmax(_outputs, dim=-1))
        outputs = _outputs + res_outputs
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            stable_model_logits = self.stable_model(buf_inputs)
            plastic_model_logits = self.plastic_model(buf_inputs)

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            _buf_outputs = self.net(buf_inputs)
            l_cons = torch.mean(self.consistency_loss(_buf_outputs, ema_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

            _buf_outputs = self.net(buf_inputs)
            with torch.no_grad():
                buf_res_outputs = self.CBA(F.softmax(_buf_outputs, dim=-1))
            buf_outputs = _buf_outputs + buf_res_outputs
            ce_loss = self.loss(buf_outputs, buf_labels)
            loss += ce_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

        return loss.item()

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.params(), self.net.params()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.params(), self.net.params()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def cba_updating(self, inputs, labels, buf_inputs, buf_labels):
        # 1. copy the model to meta models
        if self.args.backbone == 'resnet18-meta':
            meta_model = resnet18_meta(self.num_cls).to(self.device)
        meta_model.load_state_dict(self.net.state_dict())

        # 2. one step updating virtually
        loss = 0

        stable_model_logits = self.stable_model(buf_inputs)
        plastic_model_logits = self.plastic_model(buf_inputs)

        stable_model_prob = F.softmax(stable_model_logits, 1)
        plastic_model_prob = F.softmax(plastic_model_logits, 1)

        label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
        sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
        sel_idx = sel_idx.unsqueeze(1)

        ema_logits = torch.where(
            sel_idx,
            stable_model_logits,
            plastic_model_logits,
        )

        _buf_outputs = meta_model(buf_inputs)
        loss += self.args.reg_weight * torch.mean(self.consistency_loss(_buf_outputs, ema_logits.detach()))

        _outputs = meta_model(inputs)
        _buf_outputs = meta_model(buf_inputs)
        res_outputs = self.CBA(F.softmax(_outputs.detach(), dim=-1))
        buf_res_outputs = self.CBA(F.softmax(_buf_outputs.detach(), dim=-1))
        outputs = _outputs + res_outputs
        buf_outputs = _buf_outputs + buf_res_outputs

        loss += self.loss(outputs, labels.long())
        loss += self.loss(buf_outputs, buf_labels.long())

        meta_model.zero_grad()
        grads = torch.autograd.grad(loss, meta_model.fc.params(), create_graph=True)
        meta_model.fc.update_params(lr_inner=self.opt.param_groups[0]['lr'], source_params=grads)
        del grads

        # 3. update bias corrector by meta set
        buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
        _buf_outputs = meta_model(buf_inputs)
        loss_outer = self.loss(_buf_outputs, buf_labels.long())

        self.opt_cba.zero_grad()
        loss_outer.backward()
        self.opt_cba.step()


