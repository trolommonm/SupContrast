import torch
from torchvision import transforms, datasets
from data_aug import ScaleTransform, GaussianBlur, TwoCropTransform
from dommainnet_dataset import DomainNetDataset
from kaokore_dataset import Kaokore


def set_loader(opt, method):
    assert method == "supcon" or method == "ce" or method == "linear", "Invalid method!"

    train_transform, val_transform = get_augmentations(opt)
    if method == "supcon":
        train_transform = TwoCropTransform(train_transform)

    if opt.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == "domainnet":
        train_dataset = DomainNetDataset(annotations_file="DomainNet/train_combined.txt", img_dir="DomainNet/combined/",
                                         transform=train_transform)
        val_dataset = DomainNetDataset(annotations_file="DomainNet/test_combined.txt", img_dir="DomainNet/combined/",
                                       transform=val_transform)
    elif opt.dataset == "dtd":
        # merge the train and val to form the train set
        train_dataset = datasets.DTD(root=opt.data_folder,
                                     split="train",
                                     transform=train_transform,
                                     download=True)
        val_dataset = datasets.DTD(root=opt.data_folder,
                                   split="val",
                                   transform=train_transform,
                                   download=True)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

        val_dataset = datasets.DTD(root=opt.data_folder,
                                   split="test",
                                   transform=val_transform,
                                   download=True)
    elif opt.dataset == "svhn":
        train_dataset = datasets.SVHN(root=opt.data_folder,
                                      split="train",
                                      transform=train_transform,
                                      download=True)
        val_dataset = datasets.SVHN(root=opt.data_folder,
                                    split="test",
                                    transform=val_transform,
                                    download=True)
    elif opt.dataset == "kaokore":
        train_dataset = Kaokore(root="kaokore_v1.1",
                                split="train",
                                transform=train_transform)
        val_dataset = Kaokore(root="kaokore_v1.1",
                              split="test",
                              transform=val_transform)
    elif opt.dataset == "flowers102":
        train_dataset = datasets.Flowers102(root=opt.data_folder,
                                            split="train",
                                            transform=train_transform,
                                            download=True)
        val_dataset = datasets.Flowers102(root=opt.data_folder,
                                          split="val",
                                          transform=train_transform,
                                          download=True)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

        val_dataset = datasets.Flowers102(root=opt.data_folder,
                                          split="test",
                                          transform=val_transform,
                                          download=True)
    elif opt.dataset == "aircraft":
        train_dataset = datasets.FGVCAircraft(root=opt.data_folder,
                                              split="trainval",
                                              transform=train_transform,
                                              download=True)
        val_dataset = datasets.FGVCAircraft(root=opt.data_folder,
                                            split="test",
                                            transform=val_transform,
                                            download=True)
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


def get_augmentations(opt):
    if opt.augmentation == "autoaugment":
        # AutoAugment
        train_transform = transforms.Compose([
            transforms.Resize(size=(opt.size, opt.size)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy[opt.autoaugment_policy]),
            ScaleTransform() if opt.dataset == "domainnet" else transforms.ToTensor(),
            # normalize
        ])
    elif opt.augmentation == "randaugment":
        # RandAugment
        train_transform = transforms.Compose([
            transforms.Resize(size=(opt.size, opt.size)),
            transforms.RandAugment(),
            ScaleTransform() if opt.dataset == "domainnet" else transforms.ToTensor(),
            # normalize
        ])
    elif opt.augmentation == "simaugment":
        # SimAugment
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ]),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * opt.size)),
            ScaleTransform() if opt.dataset == "domainnet" else transforms.ToTensor(),
            # normalize
        ])
    elif opt.augmentation == "none":
        train_transform = transforms.Compose([
            transforms.Resize(size=(opt.size, opt.size)),
            transforms.RandomHorizontalFlip(),
            ScaleTransform() if opt.dataset == "domainnet" else transforms.ToTensor(),
            # normalize
        ])
    else:
        raise ValueError("This should not happen; check the augmentation argument!")

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        ScaleTransform() if opt.dataset == "domainnet" else transforms.ToTensor(),
        # normalize,
    ])

    return train_transform, val_transform
