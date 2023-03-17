import argparse

from main_linear import train, validate, set_model
from data_loader import set_loader, ds_to_ncls
from util import set_optimizer

from torch.cuda.amp import GradScaler

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def main(opt):
    opt.data_folder = './datasets/'
    opt.n_cls = ds_to_ncls[opt.dataset]
    opt.augmentation = "none"
    opt.momentum = 0.9
    opt.weight_decay = 0
    opt.warm = False
    opt.print_freq = 10

    epochs = 50
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [32, 64, 128]

    opt_lr = None
    opt_bs = None
    highest_val_acc = float('-inf')
    for lr in learning_rates:
        for bs in batch_sizes:
            opt.learning_rate = lr
            opt.batch_size = bs

            # build data loader
            train_loader, val_loader = set_loader(opt, "linear", validation=True)

            # build model and criterion
            model, classifier, criterion = set_model(opt)

            # set optimizer
            optimizer = set_optimizer(opt, classifier)

            # GradScalar for amp
            scalar = GradScaler(enabled=opt.amp)

            tmp_opt_lr = None
            tmp_opt_bs = None
            tmp_highest_val_acc = float('-inf')
            for epoch in range(epochs):
                _, _, _ = train(train_loader, model, classifier, criterion, optimizer, epoch, opt, scalar)

                _, val_acc, val_mean_per_class_acc = validate(val_loader, model, classifier, criterion, opt)
                if val_mean_per_class_acc:
                    if val_mean_per_class_acc > tmp_highest_val_acc:
                        tmp_highest_val_acc = val_mean_per_class_acc
                        tmp_opt_lr = lr
                        tmp_opt_bs = bs
                else:
                    if val_acc > tmp_highest_val_acc:
                        tmp_highest_val_acc = val_acc
                        tmp_opt_lr = lr
                        tmp_opt_bs = bs

            if tmp_highest_val_acc > highest_val_acc:
                highest_val_acc = tmp_highest_val_acc
                opt_lr = tmp_opt_lr
                opt_bs = tmp_opt_bs

    print(f"Hyperparameter search for dataset: {opt.dataset} method: {opt.pretrained_method} done.")
    print(f"Highest val acc: {highest_val_acc}, optimal lr: {opt_lr}, optimal bs: {opt_bs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'dtd', 'svhn', 'kaokore', 'flowers102', 'aircraft', 'pets'],
                        help='dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop/Resize')

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained_method', type=str, default='supcon', choices=['ce', 'supcon'],
                        help='method used for pre-trained model')
    parser.add_argument('--num_classes', type=int,
                        help='number of classes used for pre-trained model if method is CE')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of workers for the data loader')
    parser.add_argument('--amp', action='store_true',
                        help='enable automatic mixed precision training')
    parser.add_argument('--no_data_parallel', action='store_true',
                        help='dont use data parallel if multiple gpus are available')
    parser.add_argument('--fine_tune', action='store_true',
                        help='whether to fine-tune or perform fixed feature evaluation')

    opt = parser.parse_args()

    main(opt)
