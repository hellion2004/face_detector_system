import cv2
import time

from camera.camera import Camera
from detection.face_detector import FaceDetector

from recognition.database import FaceDatabase
from recognition.embedder import FaceEmbedder
from recognition.recogniser import Recognizer

from registration.registrar import Registrar

from utils.fps import FPSCounter
from utils.drawing import (
    draw_face,
    draw_registration_status,
    draw_recognition
)

# ==========================
# TUNABLE PARAMETERS
# ==========================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECT_EVERY_N_FRAMES = 3
RECOGNIZE_EVERY_N_FRAMES = 15

WINDOW_NAME = "Face Recognition System"

SHOW_DETECTION_TIME = False

# ==========================


def main():

    # --------------------------
    # INIT SYSTEM
    # --------------------------

    camera = Camera(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT
    )

    detector = FaceDetector()

    database = FaceDatabase()
    embedder = FaceEmbedder()

    recognizer = Recognizer(
        database,
        embedder
    )

    registrar = Registrar(
        database,
        embedder
    )

    fps_counter = FPSCounter()

    # --------------------------
    # STATE
    # --------------------------

    faces = []
    recognized_faces = []

    frame_count = 0

    try:

        while True:

            frame = camera.read()

            if frame is None:
                print("Camera read failed")
                break

            frame_count += 1

            # ==========================
            # DETECTION (SKIPPED FRAMES)
            # ==========================

            if frame_count % DETECT_EVERY_N_FRAMES == 0:

                start = time.time()

                faces = detector.detect(frame)

                if SHOW_DETECTION_TIME:
                    print(
                        f"Detection: "
                        f"{time.time() - start:.3f}s"
                    )

            # ==========================
            # RECOGNITION (SLOW LOOP)
            # ==========================

            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:

                recognized_faces = []

                for face in faces:

                    name, score = recognizer.recognize(face)

                    recognized_faces.append((name, score))

            # ==========================
            # DRAWING + REGISTRATION
            # ==========================

            for idx, face in enumerate(faces):

                draw_face(frame, face)

                registrar.process(face)

                # Recognition overlay
                if idx < len(recognized_faces):

                    name, score = recognized_faces[idx]

                    draw_recognition(
                        frame,
                        face,
                        name,
                        score
                    )

            # ==========================
            # FPS DISPLAY
            # ==========================

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

            # ==========================
            # REGISTRATION STATUS
            # ==========================

            draw_registration_status(
                frame,
                registrar
            )

            # ==========================
            # HELP TEXT
            # ==========================

            cv2.putText(
                frame,
                "R = Register | Q = Quit",
                (20, CAMERA_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ==========================
            # SHOW WINDOW
            # ==========================

            cv2.imshow(WINDOW_NAME, frame)

            # ==========================
            # KEY HANDLING
            # ==========================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Exit requested")
                break

            if key == ord("r"):

                if not registrar.active:
                    registrar.start()

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        print("Cleaning up...")
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()