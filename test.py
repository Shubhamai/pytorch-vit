# Importing Libraries
import torch
from vit_pytorch import ViT

import dataloader
import trainer
import utils


def main():
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    _, test_loader = dataloader.mnist.load_mnist()

    # Create the model
    model = ViT(
        image_size=28,
        patch_size=7,
        channels=1,
        num_classes=10,
        dim=64,
        depth=2,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Load the model
    utils.load_model(model, model_path="./experiments/models/model.pt")

    # Test the model
    test_data = trainer.test(model, test_loader, device=device, criterion=criterion)

    # Plot the results
    utils.plot_results(test_data,save_target_dir="./experiments/results", name="test_results.jpg")


if __name__ == "__main__":

    main()
