"""Simple OpenCV check for the default webcam."""

import cv2


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open the default webcam (index 0). Check that it is connected and available.")
        camera.release()
        return 1

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read a frame from the webcam.")
                return 1

            height, width = frame.shape[:2]
            cv2.putText(
                frame,
                "TouchFree Webcam Test",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Resolution: {width} x {height}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow("TouchFree Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())