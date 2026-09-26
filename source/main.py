import os
import cv2
import Video
from detector import FaceDetector
from classifier import ExpressionClassifier


def main():
    video = Video.Video_handeler()
    detector = FaceDetector()
    classifier = ExpressionClassifier()

    SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(SOURCE_DIR, "expression_resnet18.pt")

    classifier.load(WEIGHTS_PATH)
    try:
        while True:
            try:
                frame = video.read_frame()
            except RuntimeError as error:
                print(f"Skipping this frame: {error}")
                continue

            boxes, probabilities = detector.detect(frame)

            if boxes is not None:
                for box, probability in zip(boxes, probabilities):
                    x1, y1, x2, y2 = map(int, box)

                    # draw a box around the found faces
                    video.draw_box(frame, x1, y1, x2, y2)

                    # predict the expression
                    result = classifier.predict(frame, (x1, y1, x2, y2))
                    if result is not None:
                        label, confidence = result
                        # draw text under the box for the expression
                        video.draw_label(frame, x1, y2, label, confidence)

                    print(
                        f"Face: ({x1}, {y1}) -> ({x2}, {y2}), "
                        f"confidence={probability:.3f}"
                    )

            video.display_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.close_feed()


if __name__ == "__main__":
    main()
