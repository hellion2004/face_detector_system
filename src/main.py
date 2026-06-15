import cv2
import time

from camera.camera import Camera
from detection.face_detector import FaceDetector
from utils.fps import FPSCounter
from utils.drawing import draw_face


# ==================================
# TUNABLE PARAMETERS
# ==================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECT_EVERY_N_FRAMES = 3

SHOW_DETECTION_TIME = False

WINDOW_NAME = "Face Detection"

# ==================================


camera = Camera(
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT
)

detector = FaceDetector()

fps_counter = FPSCounter()

faces = []

frame_count = 0


try:

    while True:

        frame = camera.read()

        if frame is None:
            print("Camera read failed.")
            break

        frame_count += 1

        # --------------------------
        # Run detection every N frame
        # --------------------------

        if frame_count % DETECT_EVERY_N_FRAMES == 0:

            start = time.time()

            faces = detector.detect(frame)

            if SHOW_DETECTION_TIME:

                elapsed = time.time() - start

                print(
                    f"Detection Time: "
                    f"{elapsed:.3f}s"
                )

        # --------------------------
        # Draw previous detections
        # --------------------------

        for face in faces:
            draw_face(frame, face)

        # --------------------------
        # FPS
        # --------------------------

        fps = fps_counter.update()

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # --------------------------
        # Display
        # --------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # --------------------------
        # Exit
        # --------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("Exit requested.")
            break

except KeyboardInterrupt:

    print("Keyboard Interrupt.")

finally:

    print("Cleaning up...")

    camera.release()

    cv2.destroyAllWindows()