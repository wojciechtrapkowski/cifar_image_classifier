import torch
import torchvision
import torchvision.transforms as transforms


class DatasetLoader:
    def __init__(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
