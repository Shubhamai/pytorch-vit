import torch
import torchvision

def load_mnist(
    batch_size: int = 16, num_workers: int = 4, save_path: str = "data"
) -> torch.utils.data.DataLoader:
    """Load the MNIST data and returns the train and test dataloaders. The data is downloaded if it does not exist.

    Args:
        batch_size (int): The batch size.
        save_path (str): The path to save the data to.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    # Load the data
    train_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=save_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load the test data
    test_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=save_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
