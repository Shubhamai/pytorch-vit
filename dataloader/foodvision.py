""" Most of the code here is simply carried from the https://github.com/mrdbourke/pytorch-deep-learning . 
All the credits go to the original author - mrdbourke
The original code is licensed under the MIT License.

The file contains code to creating dataloader for food vision dataset. 
"""

import os
import zipfile
from pathlib import Path

import requests
import torch
from torchvision import datasets, transforms


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.

    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path(destination)
    image_path = data_path / "foodvision"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)

    return image_path


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
):
    """Creates training and testing DataLoaders.
  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.
  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


def load_foodvision(
    batch_size: int = 16,
    image_size: int = 28,
    num_workers: int = 4,
    save_path: str = "data",
) -> torch.utils.data.DataLoader:
    """Load the food vision data and returns the train and test dataloaders. The data is downloaded if it does not exist.

    Args:
        batch_size (int): The batch size.
        image_size (int): The image size.
        num_workers (int): The number of workers to use for the dataloader.
        save_path (str): The path to save the data to.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination=save_path,
    )

    # Load the data

    # Setup directories
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Setup ImageNet normalization levels (turns all images into similar distribution as ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Create transform pipeline manually
    manual_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )

    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,  # use manually created transforms
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader
