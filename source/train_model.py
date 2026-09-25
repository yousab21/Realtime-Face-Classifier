import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm

from classifier import ExpressionClassifier

# resolved from this file's location, not the caller's current working
# directory, so the script gives the same result whether it's run as
# `python3 train_model.py` from inside source/ or `python3 ./source/train_model.py`
# from the project root
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SOURCE_DIR, "..", "data")
DEFAULT_SAVE_PATH = os.path.join(SOURCE_DIR, "expression_resnet18.pt")


def transform_data():
    # resnet expects 224x224, 3-channel input normalized to imagenet's
    # per-channel statistics. fer2013 is grayscale, so the single
    # channel is duplicated into three identical ones to match the
    # shape resnet's first conv layer was built for
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_data(data_dir, transform, batch_size, val_split=0.15, seed=42):
    # ImageFolder infers one class per subdirectory and assigns
    # numeric labels alphabetically, matching the "angry", "disgust",
    # etc. folder layout under train/ and test/
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=transform
    )

    # carve a validation slice out of the training data. the test set
    # is intentionally left untouched here, so it stays a single
    # honest, unseen-until-the-end measurement rather than something
    # used to make decisions during training
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    # a fixed seed makes the split reproducible, so if you rerun with
    # different hyperparameters, differences in the resulting metrics
    # reflect the hyperparameter change, not a different random split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------
        model.train()

        running_train_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_train_loss += batch_loss * images.size(0)

            progress_bar.set_postfix(batch_loss=f"{batch_loss:.4f}")

        # Average training loss across the entire training set.
        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        model.eval()

        running_val_loss = 0.0
        correct_predictions = 0

        # No gradients are needed during validation.
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                running_val_loss += loss.item() * images.size(0)

                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()

        # Average validation loss across the entire validation set.
        val_loss = running_val_loss / len(val_loader.dataset)

        # Validation accuracy.
        val_accuracy = correct_predictions / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- train loss: {train_loss:.4f} "
            f"- val loss: {val_loss:.4f} "
            f"- val accuracy: {val_accuracy:.4f}"
        )

    return train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, loss_fn, device):
    # eval mode changes batchnorm/dropout behavior to use fixed,
    # learned statistics instead of per-batch ones
    model.eval()

    running_loss = 0.0
    correct_predictions = 0

    # disables gradient tracking, since no backward pass happens here
    # this reduces memory use during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # the predicted class is whichever output index has the
            # highest raw score (logit) for each image in the batch
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()

    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = correct_predictions / len(test_loader.dataset)

    print(f"Test loss: {avg_loss:.4f} - Test accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def save_results(model, save_path):
    # saving state_dict (weights only) rather than the full model
    # object keeps the file independent of this script's code
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained weights to {save_path}")


def clear_trained_parameters(model):
    # reinitializes the classification head with fresh random weights,
    # discarding anything learned so far. useful between hyperparameter
    # trials when reusing the same model object rather than rebuilding
    # the whole network (and re-downloading pretrained weights) each time
    in_features = model.fc.in_features
    out_features = model.fc.out_features

    device = next(model.parameters()).device
    model.fc = nn.Linear(in_features, out_features).to(device)

    return model


def main(
    num_epochs,
    learning_rate,
    beta1,
    beta2,
    batch_size=32,
    data_dir=DEFAULT_DATA_DIR,
    save_path=DEFAULT_SAVE_PATH,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    transform = transform_data()

    train_loader, val_loader, test_loader, class_names = load_data(
        data_dir,
        transform,
        batch_size,
    )

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    classifier = ExpressionClassifier(num_classes=len(class_names))

    classifier.model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [
            {
                "params": classifier.model.layer4.parameters(),
                "lr": 1e-5,
            },
            {
                "params": classifier.model.fc.parameters(),
                "lr": learning_rate,
            },
        ],
        betas=(beta1, beta2),
    )
    train_losses, val_losses, val_accuracies = train_model(
        classifier.model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        num_epochs,
    )

    evaluate_model(
        classifier.model,
        test_loader,
        loss_fn,
        device,
    )

    save_results(
        classifier.model,
        save_path,
    )

    return train_losses, val_losses, val_accuracies


if __name__ == "__main__":
    # hyperparameters are set here in one place, so trying a new
    # combination means editing these values rather than hunting
    # through the functions above
    main(
        num_epochs=10,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        batch_size=32,
    )
