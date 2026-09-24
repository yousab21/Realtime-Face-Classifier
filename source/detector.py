import torch
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect(self, frame):
        if frame is None:
            raise ValueError("Cannot detect faces in a None frame")

        boxes, probabilities = self.mtcnn.detect(frame)

        if boxes is None:
            return [], []

        height, width = frame.shape[:2]

        valid_boxes = []
        valid_probabilities = []

        for box, probability in zip(boxes, probabilities):
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            if x1 >= x2 or y1 >= y2:
                continue

            valid_boxes.append((x1, y1, x2, y2))
            valid_probabilities.append(probability)

        return valid_boxes, valid_probabilities
