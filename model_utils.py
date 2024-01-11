import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random


def multi_class_dice_coeff(true, logits, eps=1e-7):
    """Computes the Sørensen-Dice coefficient for multi-class.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_coeff: the Sørensen-Dice coefficient.
    """
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_coeff = (2.0 * intersection / (cardinality + eps)).mean()
    return dice_coeff

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen-Dice loss, which is 1 minus the Dice coefficient.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen-Dice loss.
    """
    return 1 - multi_class_dice_coeff(true, logits, eps)

def evaluate(model, val_loader, device, criterion):
    """Evaluates the model on the validation set.
    Args:
        model: The trained model.
        val_loader: DataLoader for the validation dataset.
        device: The device to run the model on.
        criterion: The loss function used for evaluation.
    Returns:
        avg_val_score: Average validation Dice score.
        avg_val_loss: Average validation loss.
    """
    model.eval()
    val_loss = 0
    val_score = 0
    with torch.no_grad():
        for batch in val_loader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.long)

            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks)
            loss += dice_loss(true_masks, masks_pred)

            val_loss += loss.item()
            val_score += multi_class_dice_coeff(true_masks, masks_pred).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_score = val_score / len(val_loader)
    return avg_val_score, avg_val_loss

def test_model(model, device, val_loader, epoch, parent_folder):
    """Visualizes predictions for one random test image.
    Args:
        model: The trained model.
        device: The device to run the model on.
        val_loader: DataLoader for the validation dataset.
        epoch: Current epoch number.
        parent_folder: Folder where to save the visualizations.
    """
    model.eval()
    with torch.no_grad():
        # Get one batch from the DataLoader
        batch = next(iter(val_loader))
        # Unpack the batch
        images, true_masks = batch['image'], batch['mask']

        # Select the first image and mask from the batch
        image = images[0].unsqueeze(0).to(device)
        mask = true_masks[0].unsqueeze(0).to(device)

        # Generate the model's prediction
        output = model(image)
        output = torch.argmax(output, dim=1)

        # Plot and save the image, true mask, and predicted mask
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.cpu().numpy()[0].transpose(1, 2, 0))
        ax[0].set_title('Input Image')
        ax[1].imshow(mask.cpu().numpy()[0], cmap='gray')
        ax[1].set_title('True Mask')
        ax[2].imshow(output.cpu().numpy()[0], cmap='gray')
        ax[2].set_title('Predicted Mask')
        plt.savefig(f"{parent_folder}/epoch_{epoch}_visualization.png")
        plt.close()


def test_model_post_training(model, device, val_loader, epoch, sample_size=50, parent_folder="output"):
    """Evaluates the model's performance on the validation set after training.
    Visualizes the model's predictions for a subset of the validation data.
    
    Args:
        model: The trained model.
        device: The device to run the model on.
        val_loader: DataLoader for the validation dataset.
        epoch: The current epoch number.
        sample_size: Number of samples to visualize.
        parent_folder: Directory where the output images will be saved.
    """
    model.eval()
    images_so_far = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device)
            true_masks = true_masks.to(device)
            outputs = model(images)

            for j in range(images.size()[0]):
                if images_so_far < sample_size:
                    images_so_far += 1
                    plt.figure(figsize=(10, 10))

                    plt.subplot(1, 3, 1)
                    plt.imshow(images[j].cpu().permute(1, 2, 0))
                    plt.title('Original Image')

                    plt.subplot(1, 3, 2)
                    plt.imshow(true_masks[j].cpu().squeeze(), cmap='gray')
                    plt.title('True Mask')

                    plt.subplot(1, 3, 3)
                    predicted_mask = torch.argmax(outputs[j], dim=0)
                    plt.imshow(predicted_mask.cpu(), cmap='gray')
                    plt.title('Predicted Mask')

                    plt.savefig(f"{parent_folder}/post_training_sample_{images_so_far}_epoch_{epoch}.png")
                    plt.close()
                else:
                    break
            if images_so_far >= sample_size:
                break


def plot_training_history(parent_folder, epoch_losses, train_scores, val_losses, val_scores):
    """
    Plots the training history of the model.

    Args:
    epoch_losses (list): List of training losses per epoch.
    train_scores (list): List of training Dice scores per epoch.
    val_losses (list): List of validation losses per epoch.
    val_scores (list): List of validation Dice scores per epoch.
    """

    epochs = range(1, len(epoch_losses) + 1)

    # Creating two figures with two subplots each
    # Figure 1: Losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_scores, label='Training Dice Score')
    plt.plot(epochs, val_scores, label='Validation Dice Score')
    plt.title('Training and Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{parent_folder}/training_progress.png")
    plt.close()