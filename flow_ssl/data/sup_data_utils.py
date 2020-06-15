import torch
import torchvision
import os
from flow_ssl.data.nlp_datasets import AG_News
from flow_ssl.data.image_datasets import SVHN_
from flow_ssl.data.image_datasets import OldInterface
from flow_ssl.data.transfer_datasets import TransferCIFAR,TransferSVHN, TransferCelebA
import torchvision.transforms as transforms
import numpy as np


def make_sup_data_loaders(
        path, 
        batch_size, 
        num_workers, 
        transform_train, 
        transform_test, 
        use_validation=True, 
        val_size=5000, 
        shuffle_train=True,
        dataset="cifar10",
        only_class=None
        ):

    
    if dataset == "notmnist":
        test_set = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
        return None, test_loader, 10

    if dataset == "numpy":
        arr = np.load(path)
        x_train, x_test = arr['train'], arr['test']
        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_train = torch.zeros(x_train.shape[:1])
        y_test = torch.zeros(x_test.shape[:1])

        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

        test_set = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        return train_loader, test_loader, 1


    if dataset == "celeba":
        image_size = 32
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(path, 'train'),
            transform=transforms.Compose([
               transforms.Resize(image_size),
               transforms.CenterCrop(image_size),
               transforms.ToTensor(),
        ]))

        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(path, 'test'),
            transform=transforms.Compose([
               transforms.Resize(image_size),
               transforms.CenterCrop(image_size),
               transforms.ToTensor(),
        ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, test_loader, 0

    download = True
    if dataset.lower() == "svhn":
        ds = SVHN_
    elif dataset.lower() == "ag_news":
        ds = AG_News
        download=False
    elif dataset.lower() == "fashionmnist":
        ds = getattr(torchvision.datasets, 'FashionMNIST')

    # transfer feature datasets
    elif dataset.lower() == "cifar10_transfer":
        ds = TransferCIFAR
        download = False
    elif dataset.lower() == "svhn_transfer":
        ds = TransferSVHN
        download = False
    elif dataset.lower() == "celeba_transfer":
        ds = TransferCelebA
        download = False
    else:
        ds = getattr(torchvision.datasets, dataset.upper())

    train_set = ds(root=path, train=True, download=download, transform=transform_train)

    if not ((hasattr(train_set, "train_data") or hasattr(train_set, "test_data"))):
        ds_base = ds
        ds = lambda *args, **kwargs: OldInterface(ds_base(*args, **kwargs))
        train_set = ds(root=path, train=True, download=download, transform=transform_train)

    num_classes = max(train_set.train_labels) + 1

    if use_validation:
        print("Using train (" + str(len(train_set.train_data)-val_size) + 
              ") + validation (" +str(val_size)+ ")")
        train_set.train_data = train_set.train_data[:-val_size]
        train_set.train_labels = train_set.train_labels[:-val_size]

        test_set = ds(root=path, train=True, download=download, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-val_size:]
        test_set.test_labels = test_set.train_labels[-val_size:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        test_set = ds(root=path, train=False, download=download, transform=transform_test)

    if only_class is not None:
        train_idx = train_set.train_labels==only_class
        test_idx = test_set.test_labels==only_class
        train_set.data = train_set.data[train_idx]
        train_set.targets = train_set.targets[train_idx]
        test_set.data = test_set.data[test_idx]
        test_set.targets = test_set.targets[test_idx]

    train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            )
    test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, test_loader, num_classes
