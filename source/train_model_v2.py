import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm

from classifier import ExpressionClassifier


# ============================================================
# Paths
# ============================================================

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR = os.path.join(
    SOURCE_DIR,
    "..",
    "data",
)

DEFAULT_SAVE_PATH = os.path.join(
    SOURCE_DIR,
    "expression_resnet18.pt",
)

DEFAULT_BEST_SAVE_PATH = os.path.join(
    SOURCE_DIR,
    "expression_resnet18_best.pt",
)


# ============================================================
# Data transforms
# ============================================================


def train_transform_data():
    """
    Transformations used only for training.

    The random augmentations help prevent the model from
    memorizing the exact appearance/position of training faces.
    """

    return transforms.Compose(
        [
            # FER images are grayscale, but ResNet expects 3 channels.
            transforms.Grayscale(num_output_channels=3),
            # ResNet expects 224x224 input.
            transforms.Resize((224, 224)),
            # ------------------------------------------------
            # Data augmentation
            # ------------------------------------------------
            # Expressions should remain valid when mirrored.
            transforms.RandomHorizontalFlip(p=0.5),
            # Small translations, rotations and scale changes.
            # Keep these conservative because aggressive
            # transformations can change facial expressions.
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            # ImageNet normalization because we use
            # ImageNet-pretrained ResNet18.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def validation_transform_data():
    """
    Transformations used for validation and testing.

    No random augmentation here because validation/test
    should measure the model on the actual images.
    """

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ============================================================
# Data loading
# ============================================================


def load_data(
    data_dir,
    batch_size,
    val_split=0.15,
    seed=42,
):
    """
    Load train/test datasets and create a validation split.

    Important:
    The training subset receives random augmentation while
    the validation subset does not.
    """

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    # --------------------------------------------------------
    # Create the base dataset without a transform.
    # --------------------------------------------------------

    full_train_dataset = datasets.ImageFolder(train_path)

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=validation_transform_data(),
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # random_split returns Subset objects pointing to the
    # same underlying ImageFolder dataset.
    #
    # Therefore we create separate datasets and use the
    # selected indices for train/validation.
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform_data(),
    )

    val_dataset = datasets.ImageFolder(
        train_path,
        transform=validation_transform_data(),
    )

    train_subset = torch.utils.data.Subset(
        train_dataset,
        train_subset.indices,
    )

    val_subset = torch.utils.data.Subset(
        val_dataset,
        val_subset.indices,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_train_dataset.classes,
    )


# ============================================================
# Training
# ============================================================


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    num_epochs,
    save_path,
    patience=3,
):
    """
    Train the model with:

    - validation monitoring
    - early stopping
    - best-model checkpointing
    - ReduceLROnPlateau scheduler
    """

    train_losses = []
    val_losses = []
    val_accuracies = []

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(num_epochs):
        # ====================================================
        # Training
        # ====================================================

        model.train()

        running_train_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )

        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = loss_fn(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_loss = loss.item()

            running_train_loss += batch_loss * images.size(0)

            progress_bar.set_postfix(batch_loss=f"{batch_loss:.4f}")

        train_loss = running_train_loss / len(train_loader.dataset)

        train_losses.append(train_loss)

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        running_val_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                outputs = model(images)

                loss = loss_fn(
                    outputs,
                    labels,
                )

                running_val_loss += loss.item() * images.size(0)

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                correct_predictions += (predictions == labels).sum().item()

        val_loss = running_val_loss / len(val_loader.dataset)

        val_accuracy = correct_predictions / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ====================================================
        # Logging
        # ====================================================

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- train loss: {train_loss:.4f} "
            f"- val loss: {val_loss:.4f} "
            f"- val accuracy: {val_accuracy:.4f} "
            f"- lr: {current_lr:.2e}"
        )

        # ====================================================
        # Best checkpoint
        #
        # We use validation loss because your current training
        # shows validation loss becoming worse even while
        # accuracy stays relatively flat.
        # ====================================================

        if val_accuracy > best_val_accuracy:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(
                f"  -> New best model saved "
                f"(val loss: {val_loss:.4f}, "
                f"val accuracy: {val_accuracy:.4f})"
            )

        else:
            epochs_without_improvement += 1

            print(f"  -> No improvement ({epochs_without_improvement}/{patience})")

        # ====================================================
        # Early stopping
        # ====================================================

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")

            break

    # ========================================================
    # Restore best model
    # ========================================================

    print(
        f"\nLoading best model "
        f"(val loss: {best_val_loss:.4f}, "
        f"val accuracy: {best_val_accuracy:.4f})"
    )

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True,
        )
    )

    return (
        train_losses,
        val_losses,
        val_accuracies,
    )


# ============================================================
# Test evaluation
# ============================================================


def evaluate_model(
    model,
    test_loader,
    loss_fn,
    device,
    class_names,
):
    """
    Evaluate the model on the test set.

    Prints:
    - Overall test loss
    - Overall test accuracy
    - Accuracy for each individual class
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    num_classes = len(class_names)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = loss_fn(
                outputs,
                labels,
            )

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            # Overall accuracy
            correct_predictions += (predictions == labels).sum().item()

            # ------------------------------------------------
            # Per-class accuracy
            # ------------------------------------------------

            for class_index in range(num_classes):
                class_mask = labels == class_index

                class_total[class_index] += class_mask.sum().item()

                class_correct[class_index] += (
                    (predictions[class_mask] == labels[class_mask]).sum().item()
                )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    avg_loss = running_loss / len(test_loader.dataset)

    accuracy = correct_predictions / len(test_loader.dataset)

    print(
        f"\nTest loss: {avg_loss:.4f}"
        f" - Test accuracy: {accuracy:.4f}"
        f" ({accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    print("\nPer-class test accuracy:")

    for class_index, class_name in enumerate(class_names):
        if class_total[class_index] > 0:
            class_accuracy = class_correct[class_index] / class_total[class_index]

            print(
                f"  {class_name:<10} "
                f"{class_accuracy:.4f} "
                f"({class_accuracy * 100:.2f}%) "
                f"[{class_correct[class_index]}/"
                f"{class_total[class_index]}]"
            )

        else:
            print(f"  {class_name:<10} N/A (no samples)")

    return avg_loss, accuracy


# ============================================================
# Save results
# ============================================================


def save_results(
    model,
    save_path,
):
    """
    Save the final best model.
    """

    torch.save(
        model.state_dict(),
        save_path,
    )

    print(f"Saved trained weights to {save_path}")


# ============================================================
# Main training function
# ============================================================


def main(
    num_epochs=20,
    learning_rate=3e-4,
    layer4_learning_rate=3e-5,
    layer3_learning_rate=3e-5,
    weight_decay=1e-4,
    batch_size=32,
    beta1=0.9,
    beta2=0.999,
    patience=4,
    data_dir=DEFAULT_DATA_DIR,
    save_path=DEFAULT_SAVE_PATH,
):
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    print(f"Classes: {class_names}")

    print(f"Training samples: {len(train_loader.dataset)}")

    print(f"Validation samples: {len(val_loader.dataset)}")

    print(f"Test samples: {len(test_loader.dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    classifier = ExpressionClassifier(num_classes=len(class_names))

    model = classifier.model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --------------------------------------------------------
    # Optimizer
    #
    # Layer4 receives a smaller learning rate because it
    # contains pretrained ImageNet features.
    #
    # The new classification head gets a larger LR because
    # its weights start from scratch.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer3.parameters(),
                "lr": layer3_learning_rate,
            },
            {
                "params": model.layer4.parameters(),
                "lr": layer4_learning_rate,
            },
            {
                "params": model.fc.parameters(),
                "lr": learning_rate,
            },
        ],
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    #
    # If validation loss stops improving, reduce LR.
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_losses, val_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs,
        save_path=save_path,
        patience=patience,
    )

    # --------------------------------------------------------
    # Test using BEST validation model
    # --------------------------------------------------------

    evaluate_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        class_names=class_names,
    )
    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    save_results(
        model=model,
        save_path=save_path,
    )

    return (
        train_losses,
        val_losses,
        val_accuracies,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main(
        num_epochs=20,
        # Classification head LR
        learning_rate=3e-4,
        # ResNet18 layer4 and layer3 LR
        layer4_learning_rate=3e-5,
        layer3_learning_rate=3e-5,
        # AdamW regularization
        weight_decay=1e-4,
        batch_size=32,
        beta1=0.9,
        beta2=0.999,
        # Stop after 4 epochs without validation
        # loss improvement.
        patience=4,
    )
