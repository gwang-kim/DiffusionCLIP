from .AFHQ_dataset import get_afhq_dataset
from .CelebA_HQ_dataset import get_celeba_dataset
from .LSUN_dataset import get_lsun_dataset
from torch.utils.data import DataLoader
from .IMAGENET_dataset import get_imagenet_dataset

def get_dataset(dataset_type, dataset_paths, config, target_class_num=None, gender=None):
    if dataset_type == 'AFHQ':
        train_dataset, test_dataset = get_afhq_dataset(dataset_paths['AFHQ'], config)
    elif dataset_type == "LSUN":
        train_dataset, test_dataset = get_lsun_dataset(dataset_paths['LSUN'], config)
    elif dataset_type == "CelebA_HQ":
        train_dataset, test_dataset = get_celeba_dataset(dataset_paths['CelebA_HQ'], config)
    elif dataset_type == "IMAGENET":
        train_dataset, test_dataset = get_imagenet_dataset(dataset_paths['IMAGENET'], config, class_num=target_class_num)
    else:
        raise ValueError

    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, bs_train=1, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs_train,
        drop_last=True,
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        sampler=None,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {'train': train_loader, 'test': test_loader}


