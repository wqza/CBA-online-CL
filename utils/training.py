# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from utils.metrics import *
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys

from copy import deepcopy
import torch.nn.functional as F


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset,
             last=False, returnt=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                if returnt == 'CBA':
                    res_outputs = model.CBA(F.softmax(outputs, dim=-1))
                    outputs = outputs + res_outputs

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    # save args
    save_path = os.path.join(os.getcwd(), 'visualization', model.NAME + '-' + dataset.NAME + '-' + args.exp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'record.txt'), 'a') as f:
        for arg in vars(args):
            f.write('{}:\t{}\n'.format(arg, getattr(args, arg)))

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME != 'scr':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    model.n_cls_per_task = dataset.N_CLASSES_PER_TASK
    model.total_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    print(file=sys.stderr)
    all_accuracy_cls, all_accuracy_tsk = [], []
    all_forward_cls, all_forward_tsk = [], []
    all_backward_cls, all_backward_tsk = [], []
    all_forgetting_cls, all_forgetting_tsk = [], []
    all_acc_auc_cls, all_acc_auc_tsk = [], []
    if hasattr(model, 'CBA'):
        all_CBA_accuracy_cls, all_CBA_accuracy_tsk = [], []
    for t in range(dataset.N_TASKS):
        model.current_task = t
        model.seen_classes = (t + 1) * dataset.N_CLASSES_PER_TASK
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        all_acc_auc_cls.append([])
        all_acc_auc_tsk.append([])

        scheduler = dataset.get_scheduler(model, args)

        for epoch in range(model.args.n_epochs):
            model.epoch = epoch  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for i, data in enumerate(train_loader):
                if model.args.n_epochs == 1 and i == len(train_loader) - 1:
                    # End the training before the last iteration.
                    # That is because we find that the last few samples would hurt the network training,
                    # which leads to lower performance.
                    continue

                model.ii = i
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                # anytime inference
                if model.NAME != 'icarl' and model.args.n_epochs == 1 and args.dataset != 'seq-imgnet1k' and model.NAME != 'scr':
                    if i % 5 == 0:
                        accs = evaluate(deepcopy(model), dataset)
                        all_acc_auc_cls[t].append(accs[0])
                        all_acc_auc_tsk[t].append(accs[1])

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        print('class-il:', accs[0], '\ntask-il:', accs[1])

        # record the results
        all_accuracy_cls.append(accs[0])
        all_accuracy_tsk.append(accs[1])

        # print the fwt, bwt, forgetting
        if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME != 'scr':
            fwt = forward_transfer(results, random_results_class)
            fwt_mask_classes = forward_transfer(results_mask_classes, random_results_task)
            bwt = backward_transfer(results)
            bwt_mask_classes = backward_transfer(results_mask_classes)
            forget = forgetting(results)
            forget_mask_classes = forgetting(results_mask_classes)
            print('Forward: class-il: {}\ttask-il:{}'.format(fwt, fwt_mask_classes))
            print('Backward: class-il: {}\ttask-il:{}'.format(bwt, bwt_mask_classes))
            print('Forgetting: class-il: {}\ttask-il:{}'.format(forget, forget_mask_classes))

            # record the results
            all_forward_cls.append(fwt)
            all_forward_tsk.append(fwt_mask_classes)
            all_backward_cls.append(bwt)
            all_backward_tsk.append(bwt_mask_classes)
            all_forgetting_cls.append(forget)
            all_forgetting_tsk.append(forget_mask_classes)

        if hasattr(model, 'CBA'):
            print('\n************************ Results of the CBA: ************************')
            accs_bias = evaluate(model, dataset, returnt='CBA')
            print_mean_accuracy(np.mean(accs_bias, axis=1), t + 1, dataset.SETTING)
            print('class-il:', accs_bias[0], '\ntask-il:', accs_bias[1])
            print('***********************************************************************************')

            # record the results
            all_CBA_accuracy_cls.append(accs_bias[0])
            all_CBA_accuracy_tsk.append(accs_bias[1])

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

        np.set_printoptions(suppress=True)

    # record the results
    with open(os.path.join(save_path, 'record.txt'), 'a') as f:
        f.write('\n== 1. Acc:\n==== 1.1. Class-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write(str(all_accuracy_cls[t]).strip('[').strip(']') + '\n')
        f.write('\n==== 1.2. Task-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write(str(all_accuracy_tsk[t]).strip('[').strip(']') + '\n')

        f.write('\n== 2. Forward:')
        f.write('\n==== 2.1. Class-IL:\n' + str(all_forward_cls).strip('[').strip(']'))
        f.write('\n==== 2.2. Task-IL:\n' + str(all_forward_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 3. Backward:')
        f.write('\n==== 3.1. Class-IL:\n' + str(all_backward_cls).strip('[').strip(']'))
        f.write('\n==== 3.2. Task-IL:\n' + str(all_backward_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 4. Forgetting:')
        f.write('\n==== 4.1. Class-IL:\n' + str(all_forgetting_cls).strip('[').strip(']'))
        f.write('\n==== 4.2. Task-IL:\n' + str(all_forgetting_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 5. Acc_auc:\n==== 5.1. Class-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write('\nTask {}:\n'.format(t + 1))
            avg_acc_cls, avg_acc_tsk = [], []
            for acc_cls, acc_tsk in zip(all_acc_auc_cls[t], all_acc_auc_tsk[t]):
                avg_acc_cls.append(np.mean(acc_cls))
                avg_acc_tsk.append(np.mean(acc_tsk))
                f.write(str(acc_cls).strip('[').strip(']') + ' - ' + str(np.mean(acc_cls)) + '\n')

        f.write('\nACC_AUC_cls = {}:\n'.format(np.mean(avg_acc_cls)))
        f.write('ACC_AUC_tsk = {}:\n'.format(np.mean(avg_acc_tsk)))

        if hasattr(model, 'CBA') and len(all_CBA_accuracy_cls) > 0:
            f.write('\n== 5. CBA Acc:\n==== 5.1. Class-IL:\n')
            for t in range(dataset.N_TASKS):
                f.write(str(all_CBA_accuracy_cls[t]).strip('[').strip(']') + '\n')
            f.write('\n==== 5.2. Task-IL:\n')
            for t in range(dataset.N_TASKS):
                f.write(str(all_CBA_accuracy_tsk[t]).strip('[').strip(']') + '\n')

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME != 'scr':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
