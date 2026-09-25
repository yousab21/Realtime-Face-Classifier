import torch
from torchvision.models import resnet18, ResNet18_Weights


class ExpressionClassifier:
    def __init__(self, num_classes=7):
        # Load ResNet18 with pretrained ImageNet weights.
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze the entire pretrained network first.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Replace the original ImageNet classification head
        # (512 -> 1000) with a new head for our emotion classes
        # (512 -> num_classes).
        self.model.fc = torch.nn.Linear(
            in_features=512,
            out_features=num_classes,
        )

        # The new classification head is trainable by default.
        for parameter in self.model.fc.parameters():
            parameter.requires_grad = True

        # unfreeze layers 3 and 4 (last 2 layer in resnet18)
        for parameter in self.model.layer4.parameters():
            parameter.requires_grad = True

        for parameter in self.model.layer3.parameters():
            parameter.requires_grad = True

    def save(self, path):
        # Save only the learned parameter values.
        torch.save(
            self.model.state_dict(),
            path,
        )

    def load(self, path):
        # Load previously saved weights.
        state_dict = torch.load(
            path,
            map_location=torch.device("cpu"),
        )

        self.model.load_state_dict(state_dict)
