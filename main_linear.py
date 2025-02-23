from __future__ import print_function

import json
import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from data_aug import GaussianBlur, ScaleTransform
# from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, SupCEResNet, LinearClassifier

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
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # augmentation
    parser.add_argument('--augmentation', type=str, default='simaugment',
                        choices=['autoaugment', 'randaugment', 'simaugment', 'none'], help='choose augmentation')
    parser.add_argument('--autoaugment_policy', required=False,
                        choices=['IMAGENET', 'CIFAR10', 'SVHN'])

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained_method', type=str, default='supcon', choices=['ce', 'supcon'],
                        help='method used for pre-trained model')
    parser.add_argument('--num_classes', type=int,
                        help='number of classes used for pre-trained model if method is CE')
    parser.add_argument('--amp', action='store_true',
                        help='enable automatic mixed precision training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--subtrial', type=str, default='0',
                        help='subtrial id for recording multiple runs of the same trial')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/{}_LinearEval/{}_models'.format(opt.pretrained_method, opt.dataset)
    opt.tb_path = './save/{}_LinearEval/{}_tensorboard'.format(opt.pretrained_method, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)
    if opt.subtrial:
        opt.model_name += "_subtrial_{}".format(opt.subtrial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if opt.augmentation == 'autoaugment':
        assert opt.autoaugment_policy is not None, \
            "Please specific the AutoAugment policy to be used for AutoAugment!"

    if opt.pretrained_method == 'ce':
        assert opt.num_classes is not None, \
            "For CE, please specify the number of classes during the pre-training process!"

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # save arguments used
    with open(os.path.join(opt.save_folder, 'args.json'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    return opt


def set_model(opt):
    if opt.pretrained_method == 'supcon':
        model = SupConResNet(name=opt.model)
    elif opt.pretrained_method == 'ce':
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)

    return model, classifier, criterion


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    opt.size = 32
    if opt.augmentation == 'autoaugment':
        # AutoAugment
        train_transform = transforms.Compose([
            transforms.Resize(size=(opt.size, opt.size)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy[opt.autoaugment_policy]),
            transforms.ToTensor(),
            # normalize
        ])
    elif opt.augmentation == 'randaugment':
        # RandAugment
        train_transform = transforms.Compose([
            transforms.Resize(size=(opt.size, opt.size)),
            transforms.RandAugment(),
            transforms.ToTensor(),
            # normalize
        ])
    elif opt.augmentation == 'simaugment':
        # SimAugment
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ]),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * opt.size)),
            transforms.ToTensor(),
            # normalize
        ])
    elif opt.augmentation == 'none':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize
        ])
    else:
        raise ValueError('This should not happen; check the augmentation argument!')

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt, scalar):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with autocast(enabled=opt.amp):
            with torch.no_grad():
                features = model.encoder(images)
        with autocast(enabled=opt.amp):
            output = classifier(features.detach())
            loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Model Summary")
    print(model)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Classifier Summary")
    print(classifier)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # GradScalar for amp
    scalar = GradScaler(enabled=opt.amp)

    # tensorboard
    logger = SummaryWriter(log_dir=opt.tb_folder)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, classifier, criterion,
                                      optimizer, epoch, opt, scalar)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, train_acc))

        # eval for one epoch
        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_file = os.path.join(opt.save_folder, 'best.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file, scalar)

        # tensorboard logger
        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('train_accuracy', train_acc, epoch)
        logger.add_scalar('val_loss', val_loss, epoch)
        logger.add_scalar('val_accuracy', val_acc, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, epoch, save_file, scalar)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
