import cv2


class Video_handeler:
    def __init__(self):
        self.camera_feed = cv2.VideoCapture(0)

        if not self.camera_feed.isOpened():
            raise RuntimeError("Could not open camera")

    def read_frame(self):
        read_successful, frame = self.camera_feed.read()

        if not read_successful:
            raise RuntimeError("Could not read frame from camera")

        return frame

    def close_feed(self):
        self.camera_feed.release()
        cv2.destroyAllWindows()

    def display_frame(self, frame):
        if frame is None:
            raise ValueError("Cannot display a None frame")

        cv2.imshow("frame", frame)

    def draw_box(self, frame, x1, y1, x2, y2):
        if frame is None:
            raise ValueError("Cannot draw on a None frame")

        if (
            x1 >= x2
            or y1 >= y2
            or x1 < 0
            or y1 < 0
            or x2 > frame.shape[1]
            or y2 > frame.shape[0]
        ):
            raise ValueError(f"Invalid box coordinates: ({x1},{y1}) to ({x2},{y2})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_label(self, frame, x1, y2, label, confidence):
        if frame is None:
            raise ValueError("Cannot draw on a None frame")

        text = f"{label} ({confidence:.2f})"

        # Positioned just below the bottom-left corner of the box (y2),
        # matching where draw_box's rectangle already ends
        text_position = (x1, y2 + 20)

        cv2.putText(
            frame,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
