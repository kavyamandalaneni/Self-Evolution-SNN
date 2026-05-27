# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(dataset_name="MNIST", batch_size=64):
    """
    Returns train/test loaders and the input size for the specified dataset.
    """
    
    # Standard transforms for small datasets
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == "MNIST":
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform_mnist)
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform_mnist)
        input_size = 28 * 28
        num_classes = 10
        
    elif dataset_name == "Fashion-MNIST":
        train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform_mnist)
        test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform_mnist)
        input_size = 28 * 28
        num_classes = 10
        
    elif dataset_name == "CIFAR-10":
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform_cifar)
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform_cifar)
        input_size = 32 * 32 * 3 # Color images
        num_classes = 10

    # =========================================================================
    # UNCOMMENT THE FOLLOWING ON A BETTER COMPUTER (RTX GPU REQUIRED)
    # =========================================================================
    # elif dataset_name == "CIFAR-100":
    #     train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=transform_cifar)
    #     test_ds = datasets.CIFAR100("./data", train=False, download=True, transform=transform_cifar)
    #     input_size = 32 * 32 * 3
    #     num_classes = 100
    #
    # elif dataset_name == "Tiny-ImageNet":
    #     # Requires manual download or specific library
    #     pass 
    # =========================================================================

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, input_size, num_classes
