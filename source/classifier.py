import torch
import torchvision.transforms as T
from torchvision.models import resnet18


class ExpressionClassifier:
    """
    Loads a fine-tuned ResNet18 expression classifier and runs
    inference on cropped face images.
    """

    # Order matches datasets.ImageFolder's alphabetical sort of
    # data/train subfolders, confirmed against the actual dataset layout
    EXPRESSION_LABELS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    def __init__(self, num_classes=7, device="cpu"):
        self.device = device

        # No pretrained ImageNet weights needed here -- load() overwrites
        # every parameter anyway, so building the bare architecture skips
        # a wasted download/initialization step
        self.model = resnet18(weights=None)

        self.model.fc = torch.nn.Linear(
            in_features=512,
            out_features=num_classes,
        )

        self.model.to(self.device)

        self.transform = self._build_transform()

    def _build_transform(self):
        # Mirrors validation_transform_data() from train_model.py exactly:
        # no augmentation at inference, but grayscale conversion is required
        # since the model only ever saw grayscale-replicated 3-channel input
        return T.Compose(
            [
                T.ToPILImage(),
                T.Grayscale(num_output_channels=3),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Puts BatchNorm/Dropout into inference mode -- required for
        # correct single-image predictions, not just a formality
        self.model.eval()

    def preprocess(self, face_crop):
        # Adds the batch dimension the model expects: [3, 224, 224] -> [1, 3, 224, 224]
        return self.transform(face_crop).unsqueeze(0).to(self.device)

    def predict(self, frame, box):
        x1, y1, x2, y2 = box
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            # Degenerate crop (e.g. box collapsed to zero width/height)
            return None

        face_tensor = self.preprocess(face_crop)

        with torch.no_grad():
            logits = self.model(face_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        label = self.EXPRESSION_LABELS[predicted_index.item()]
        return label, confidence.item()
