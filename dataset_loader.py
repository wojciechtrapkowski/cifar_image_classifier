import torch
import torchvision
import torchvision.transforms as transforms


class DatasetLoader:
    def __init__(self):
        # Use data augmentation & normalization
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        # Download the CIFAR-10 training dataset
        trainset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        # Create loader for the training set
        self.trainloader = trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=32,
            shuffle=True,
            num_workers=1,  # Number of threads for loading data
        )

        # Download the CIFAR-10 test dataset
        testset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        # Create loader for the test set
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False, num_workers=1
        )

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
