# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
import torch.nn.functional as F
np.random.seed(0)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via Meta-Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # it worth to denote this operation
    # remove batch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'batch_size':
            del parser._actions[i]
            break
    parser.add_argument('--beta', type=float, required=True,
                        help='Within-batch update beta parameter.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Across-batch update gamma parameter.')
    parser.add_argument('--batch_num', type=int, required=True,
                        help='Number of batches extracted from the buffer.')
    return parser


class Mer(ContinualModel):
    NAME = 'mer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    def __init__(self, backbone, loss, args, transform):
        super(Mer, self).__init__(backbone, loss, args, transform) # continual_model.py
        self.buffer = Buffer(self.args.buffer_size, self.device)
        assert args.batch_size == 1, 'Mer only works with batch_size=1' # why?


    
    # Perm-mnist work version
    '''
    def draw_batches(self, inp, lab):
        batches = []
        # for i in range(self.args.batch_num):
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((buf_inputs, inp.unsqueeze(0)))
            labels = torch.cat((buf_labels, torch.tensor([lab]).to(self.device)))
            batches.append((inputs.unsqueeze(0), labels.unsqueeze(0)))
            # batches.append((inputs,labels))
        else:
            batches.append((inp.unsqueeze(0), torch.tensor([lab]).unsqueeze(0).to(self.device)))
        return batches
    def observe(self, inputs, labels, not_aug_inputs,t):

        # batches = self.draw_batches(inputs, labels)    # random sample a batch data of memory buffer
        theta_A0 = self.net.get_params().data.clone()
        # print('len',len(batches))
        # for i in range(self.args.batch_num):     
        #     theta_Wi0 = self.net.get_params().data.clone()
        #     batch_inputs, batch_labels = batches[i]
        #     # print('batch_inputs', i, batch_inputs.shape)
        #     # print('batch_labels', i, batch_labels.shape)

        #     # within-batch step
        #     self.opt.zero_grad()
        #     outputs = self.net(batch_inputs)
        #     loss = self.loss(outputs, batch_labels.squeeze(-1))  # train on these batch
        #     loss.backward()
        #     self.opt.step()
        #     # within batch reptile meta-update
        #     new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
        #     self.net.set_params(new_params)

        # print('batches',batches[0][2].shape)
        for i in range(self.args.batch_num):          
            theta_Wi0 = self.net.get_params().data.clone()
            batches = self.draw_batches(inputs, labels) 
            # print('batches', len(batches)) #1
            batch_inputs, batch_labels = batches[0]
            print('batch_inputs',batch_inputs.shape)
            batch_inputs = batch_inputs.squeeze(0)
            batch_labels = batch_labels.squeeze(0)

            # print('batch_inputs',batch_inputs.shape)      #[1,11,1,28,28]
            # print('batch_inputs', i, batch_inputs.shape)
            # print('batch_labels', i, batch_labels.shape)  #[1,11]
            # print('batches3',batches[3][0].shape)
            # print('bi', batch_inputs.shape)
            # print('bl', batch_labels.shape)
            loss = 0.0
            # print('len',len(batch_inputs))
            # print('batch_labels1',batch_labels)
            for idx in range(len(batch_inputs)):
                self.opt.zero_grad()
                bx = batch_inputs[idx]
                # print('idx', idx, bx.shape )
                by = batch_labels[idx].unsqueeze(0)
                print('bx', bx.shape)
                # print('by', by,by.shape)
                prediction = self.net(bx)
                print('prediction',prediction.shape)
                # print('by', by.shape)
                loss = self.loss(prediction, by)
                loss.backward()
                self.opt.step()
            # within batch reptile meta-update
            new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
            self.net.set_params(new_params)

        # across batch reptile meta-update
        new_new_params = theta_A0 + self.args.gamma * (self.net.get_params() - theta_A0)
        self.net.set_params(new_new_params)
        self.buffer.add_data(examples=not_aug_inputs.unsqueeze(0), labels=labels)
        return loss.item()
        '''

    def draw_batches(self, inp, lab):
        batches = []
        # for i in range(self.args.batch_num):
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            # print('buf_inputs',buf_inputs.shape)
            # print('inp', inp.shape)
            inputs = torch.cat((buf_inputs, inp))
            labels = torch.cat((buf_labels, torch.tensor([lab]).to(self.device)))
            batches.append((inputs.unsqueeze(0), labels.unsqueeze(0)))
            # batches.append((inputs,labels))
        else:
            batches.append((inp.unsqueeze(0), torch.tensor([lab]).unsqueeze(0).to(self.device)))
        return batches

    # Cifar10/100 version
    def observe(self, inputs, labels, not_aug_inputs,t):
        theta_A0 = self.net.get_params().data.clone()
        # print('inputs', inputs.shape)
        # print('labels', labels.shape)
        for i in range(self.args.batch_num):          
            theta_Wi0 = self.net.get_params().data.clone()
            batches = self.draw_batches(inputs, labels) 
            batch_inputs, batch_labels = batches[0]
            # print('batch_inputs',batch_inputs.shape)
            batch_inputs = batch_inputs.squeeze(0)
            batch_labels = batch_labels.squeeze(0)
            # print('batch_inputs1',batch_inputs.shape)
            # print('batch_inputs',batch_inputs.shape)      #[1,11,1,28,28]
            # print('batch_inputs', i, batch_inputs.shape)
            # print('batch_labels', i, batch_labels.shape)  #[1,11]
            # print('batches3',batches[3][0].shape)
            # print('bi', batch_inputs.shape)
            # print('bl', batch_labels.shape)
            loss = 0.0
            # print('len',len(batch_inputs))
            # print('batch_labels1',batch_labels)
            for idx in range(len(batch_inputs)):
                self.opt.zero_grad()
                bx = batch_inputs[idx]
                if len(bx.shape) == 3:
                    bx = bx.unsqueeze(0)
                # print('idx', idx, bx.shape )
                by = batch_labels[idx].unsqueeze(0).long()
                # print('bx', bx.shape)
                # print('by', by,by.shape)
                prediction = self.net(bx)
                # print('prediction',prediction.shape)
                # print('by', by.shape)
                loss = self.loss(prediction, by)
                loss.backward()
                self.opt.step()
            # within batch reptile meta-update
            new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
            self.net.set_params(new_params)
        # across batch reptile meta-update
        new_new_params = theta_A0 + self.args.gamma * (self.net.get_params() - theta_A0)
        self.net.set_params(new_new_params)
        # print('exampels',not_aug_inputs.shape)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return loss.item()

    # variance reduction version
    # def draw_batches(self, inp, lab):
    #     batches = []
    #     # for i in range(self.args.batch_num):
    #     if not self.buffer.is_empty():
    #         buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
    #         inputs = torch.cat((buf_inputs, inp))
    #         labels = torch.cat((buf_labels, torch.tensor([lab]).to(self.device)))
    #         batches.append((inputs.unsqueeze(0), labels.unsqueeze(0)))
    #     else:
    #         batches.append((inp.unsqueeze(0), torch.tensor([lab]).unsqueeze(0).to(self.device)))
    #     return batches

    # def agreement_weights(self, grad_store, avg_grad):
    #     grad_store = torch.stack(grad_store)
    #     avg_grad = avg_grad.unsqueeze(1)
    #     # print('grad_store', grad_store.shape)
    #     # print('avg_grad', avg_grad.shape)
    #     weights_mul = grad_store.mm(avg_grad)/ (torch.norm(avg_grad,p=2)* torch.norm(grad_store,dim=1).unsqueeze(1))
    #     # print('weights_mul', weights_mul.shape)
    #     # print('weights_mul')
    #     # weights =  F.normalize(weights_mul,p=2,dim=0)
    #     weights = F.softmax(weights_mul,dim=0)
    #     # weights = weights_mul/ sum(weights_mul)
    #     return weights


    # def observe(self, inputs, labels, not_aug_inputs,t):
    #     theta_A0 = self.net.get_params().data.clone()
    #     gradient_average = 0
    #     gradient_store = []
    #     for i in range(self.args.batch_num):          
    #         theta_Wi0 = self.net.get_params().data.clone()
    #         batches = self.draw_batches(inputs, labels) 
    #         batch_inputs, batch_labels = batches[0]
    #         batch_inputs = batch_inputs.squeeze(0)
    #         batch_labels = batch_labels.squeeze(0)

    #         loss = 0.0

    #         for idx in range(len(batch_inputs)):
    #             self.opt.zero_grad()
    #             bx = batch_inputs[idx]
    #             if len(bx.shape) == 3:
    #                 bx = bx.unsqueeze(0)

    #             by = batch_labels[idx].unsqueeze(0).long()
    #             prediction = self.net(bx)
    #             loss = self.loss(prediction, by)
    #             loss.backward()
    #             self.opt.step()

    #         # within batch reptile meta-update
    #         # new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
    #         # self.net.set_params(new_params)

    #         gradient_average += self.net.get_params() - theta_Wi0
    #         gradient_store.append(self.net.get_params() - theta_Wi0)
    #         new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
    #         self.net.set_params(new_params)

    #     gradient_average = gradient_average/self.args.batch_num  # average gradient
    #     weights = self.agreement_weights(gradient_store, gradient_average)
    #     # print('weights',weights)
    #     theta_Wi0 = theta_A0
    #     for i in range(self.args.batch_num):
    #         new_params = theta_Wi0 + self.args.beta * 10 * weights[i]* gradient_store[i]
    #         # new_params = theta_Wi0 + self.args.beta * gradient_store[i]
    #         self.net.set_params(new_params)
    #         theta_Wi0 = self.net.get_params().data.clone()  # with 10 68.73%  5 69.81%

    #     # gradient_average = gradient_average/self.args.batch_num  # average gradient
    #     # weights = self.agreement_weights(gradient_store, gradient_average)
    #     # print('weights', weights)
    #     # theta_Wi0 = theta_A0
    #     # for i in range(self.args.batch_num):
    #     #     new_params = theta_Wi0 + self.args.beta * 10 * weights[i]* gradient_store[i]
    #     #     # new_params = theta_Wi0 + self.args.beta * gradient_store[i]
    #     #     self.net.set_params(new_params)
    #     #     theta_Wi0 = self.net.get_params().data.clone()  # with 10 68.73%  5 69.81%
            
    #     # across batch reptile meta-update
    #     new_new_params = theta_A0 + self.args.gamma * (self.net.get_params() - theta_A0)
    #     self.net.set_params(new_new_params)
    #     # print('exampels',not_aug_inputs.shape)
    #     self.buffer.add_data(examples=not_aug_inputs, labels=labels)
    #     return loss.item()

