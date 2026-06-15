import cv2

from camera.camera import Camera
from detection.face_detector import FaceDetector
from utils.fps import FPSCounter
from utils.drawing import draw_face


camera = Camera()

detector = FaceDetector()

fps_counter = FPSCounter()

while True:

    frame = camera.read()

    if frame is None:
        break

    faces = detector.detect(frame)

    for face in faces:
        draw_face(frame, face)

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

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Exiting...")
        break

camera.release()

cv2.destroyAllWindows()