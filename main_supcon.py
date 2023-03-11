from __future__ import print_function

import os
import sys
import argparse
import time
import math
import json

# import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.cuda.amp import autocast

from util import AverageMeter
from data_aug import ScaleTransform, GaussianBlur, TwoCropTransform
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, set_gradscalar
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from dommainnet_dataset import DomainNetDataset
from data_loader import set_loader

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='domainnet',
                        choices=['cifar10', 'cifar100', 'path', 'domainnet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop/Resize')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # augmentation
    parser.add_argument('--augmentation', type=str, default='simaugment',
                        choices=['autoaugment', 'randaugment', 'simaugment', 'stacked_randaugment'], help='choose augmentation')
    parser.add_argument('--autoaugment_policy', required=False,
                        choices=['IMAGENET', 'CIFAR10', 'SVHN'])

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--amp', action='store_true',
                        help='enable automatic mixed precision training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # resume training
    parser.add_argument('--resume', action='store_true',
                        help='resume training')
    parser.add_argument('--model_name', type=str, required=False,
                        help='model name that was created during training')
    parser.add_argument('--model_ckpt', type=str, required=False,
                        help='checkpoint to resume from')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    # if opt.batch_size > 256:
    #     opt.warm = True

    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.augmentation == 'autoaugment':
        assert opt.autoaugment_policy is not None, \
            "Please specific the AutoAugment policy to be used for AutoAugment!"

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    if opt.resume:
        assert opt.model_name is not None, "Please specify the model name that was created during training!"
        assert opt.model_ckpt is not None, "Please specify the checkpoint to resume from!"
    else:
        opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'. \
            format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                   opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)
        if opt.warm:
            opt.model_name = '{}_warm'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # save arguments used
    args_filename = 'args_resume.json' if opt.resume else 'args.json'
    with open(os.path.join(opt.save_folder, args_filename), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    return opt


def set_model(opt, ckpt=None):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if ckpt:
        state_dict = ckpt["model"]

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using Data Parallel model, detected {torch.cuda.device_count()} GPUs.")
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            if ckpt:
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict

        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    if ckpt:
        model.load_state_dict(state_dict)

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, scalar):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with autocast(enabled=opt.amp):
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == 'SupCon':
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, _ = set_loader(opt, "supcon")

    # resume training
    if opt.resume:
        model_ckpt_path = os.path.join(opt.save_folder, opt.model_ckpt)
        print(f"Resuming training, loading {model_ckpt_path}...")
        ckpt = torch.load(model_ckpt_path)
    else:
        ckpt = None

    # build model and criterion
    model, criterion = set_model(opt, ckpt)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Model Summary")
    print(model)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # build optimizer
    optimizer = set_optimizer(opt, model, ckpt)

    # GradScalar for amp
    scalar = set_gradscalar(opt, ckpt)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = SummaryWriter(log_dir=opt.tb_folder)

    # training routine
    start_epoch = ckpt["epoch"] + 1 if opt.resume else 1
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, scalar)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logger.add_scalar('loss', loss, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file, scalar)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file, scalar)


if __name__ == '__main__':
    main()
