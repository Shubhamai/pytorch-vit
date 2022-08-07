# Importing Libraries
import torch
import torch.nn as nn


def step(
    model: nn.Module,
    img: torch.Tensor,
    label: torch.Tensor,
    optimizer: torch.optim,
    criterion: torch.nn.Module,
    is_train: bool = True,
) -> dict:
    """Perform a single step of training or testing.

    Generate the predictions and calculate the loss and accuracy. If training, also perform backpropagation.

    Args:
        model (nn.Module): The model to train.
        img (torch.Tensor): The image to train on.
        label (torch.Tensor): The label to train on.
        optimizer (torch.optim): The optimizer to use.
        criterion: The loss function to use.
        train (bool): if it's in training mode, perform backpropagation, otherwise, just compute the loss and accuracy.
    Returns:
        dict: The loss and accuracy of individual step and epoch."""

    preds = model(img).squeeze()

    accuracy = torch.mean((preds.argmax(dim=1) == label).float())
    loss = criterion(preds, label)

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, accuracy


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train the model for given epochs.

    Iterate over the training set and perform a single step of training, including backpropagation.
    Store the loss and accuracy of each step. Also store the loss and accuracy of each epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The train loader.
        optimizer (torch.optim): The optimizer to use.
        criterion: The loss function to use.
        device (torch.device): The device to use.
        epoch (int): The number of epochs to train for.

    Returns:
        dict: a dict containing loss and accuracy of individual steps and epochs. For example:

        {
            "step": {
                "loss": [...],
                "accuracy": [[...]
            },
            "epoch": {
                "loss": [[...],
                "accuracy": [[...]
            }
        }
    """

    model.train()

    data = {"step": {"loss": [], "accuracy": []}, "epoch": {"loss": [], "accuracy": []}}
    for n_epoch in range(epoch):

        loss_step = []
        accuracy_step = []
        for index, (img, label) in enumerate(train_loader):

            img, label = img.to(device), label.to(device)

            loss, accuracy = step(model, img, label, optimizer, criterion)

            loss_step.append(loss.item())
            accuracy_step.append(accuracy.item())

            if index % 100 == 0:
                print(
                    f"Epoch: {n_epoch+1}/{epoch} | Batch: {index}/{len(train_loader)} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}"
                )

        data["step"]["loss"].extend(loss_step)
        data["step"]["accuracy"].extend(accuracy_step)

        data["epoch"]["loss"].append(sum(loss_step) / len(train_loader))
        data["epoch"]["accuracy"].append(sum(accuracy_step) / len(train_loader))

        print(
            f"Train Epoch: {n_epoch+1}/{epoch} | Loss: {data['epoch']['loss'][n_epoch]:.4f} | Accuracy: {data['epoch']['accuracy'][n_epoch]:.4f}"
        )

    return data


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> dict:
    """Test the model on the test set.

    Args:
        model (nn.Module): The model to test.
        test_loader (torch.utils.data.DataLoader): The test loader.
        device (torch.device): The device to use.
        criterion: The loss function to use.

    Iterate over the test set and perform a single step of testing, without backpropagation.
    Store the loss and accuracy of each step. Also store the loss and accuracy of each epoch.

    Returns:
        dict: a dict containing loss and accuracy of individual steps and epochs. For example:

        {
            "step": {
                "loss": [...],
                "accuracy": [[...]
            },
            "epoch": {
                "loss": [[...],
                "accuracy": [[...]
            }
        }
    """

    model.eval()

    data = {"step": {"loss": [], "accuracy": []}, "epoch": {"loss": [], "accuracy": []}}

    with torch.no_grad():

        for index, (img, label) in enumerate(test_loader):

            img, label = img.to(device), label.to(device)

            loss, accuracy = step(model, img, label, None, criterion, is_train=False)
            data["step"]["loss"].append(loss.item())
            data["step"]["accuracy"].append(accuracy.item())

            if index % 100 == 0:
                print(
                    f"Batch: {index}/{len(test_loader)} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}"
                )
    data["epoch"]["loss"].append(sum(data["step"]["loss"]) / len(test_loader))
    data["epoch"]["accuracy"].append(sum(data["step"]["accuracy"]) / len(test_loader))
    print(
        f"Test Loss: {data['epoch']['loss'][-1]:.4f} | Accuracy: {data['epoch']['accuracy'][-1]:.4f}"
    )

    return data
