import cv2
import Video
from detector import FaceDetector


def main():
    video = Video.Video_handeler()
    detector = FaceDetector()

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

                    video.draw_box(frame, x1, y1, x2, y2)

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
