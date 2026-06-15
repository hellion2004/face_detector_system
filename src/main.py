import cv2
import time

from camera.camera import Camera
from detection.face_detector import FaceDetector

from recognition.database import FaceDatabase
from recognition.embedder import FaceEmbedder
from recognition.recogniser import Recognizer
from recognition.recognition_manager import RecognitionManager

from registration.registrar import Registrar
from tracking.tracker import Tracker

from utils.fps import FPSCounter
from utils.drawing import draw_registration_status

# ==========================
# TUNABLE PARAMETERS
# ==========================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECT_EVERY_N_FRAMES = 3
RECOGNIZE_PERIODIC_INTERVAL = 2.0  # Seconds between re-recognition

WINDOW_NAME = "Face Recognition System"
SHOW_DETECTION_TIME = False

# ==========================


def main():

    # --------------------------
    # INIT SYSTEM
    # --------------------------

    camera = Camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    detector = FaceDetector()
    tracker = Tracker(max_age=30, distance_threshold=50.0, iou_threshold=0.3)
    
    database = FaceDatabase()
    embedder = FaceEmbedder()
    recognizer = Recognizer(database, embedder)
    
    # NEW: RecognitionManager
    recognition_manager = RecognitionManager(
        recognizer=recognizer,
        new_track_recognition=True,
        periodic_refresh_interval=RECOGNIZE_PERIODIC_INTERVAL,
        unknown_threshold=0.5,
        flickering_threshold=2
    )
    
    registrar = Registrar(database, embedder)
    fps_counter = FPSCounter()

    # --------------------------
    # STATE
    # --------------------------

    tracked_faces = []
    frame_count = 0
    detections_by_frame = {}

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
                detections = detector.detect(frame)
                
                # Convert to bboxes for tracker
                bboxes = [
                    (face.bbox[0], face.bbox[1], 
                     face.bbox[2], face.bbox[3])
                    for face in detections
                ]
                
                # Update tracker → get stable track IDs
                tracked_faces = tracker.update(bboxes)
                
                # Map track IDs to face objects
                detections_map = {}
                for det_idx, face in enumerate(detections):
                    bbox = (face.bbox[0], face.bbox[1], 
                           face.bbox[2], face.bbox[3])
                    for track_id, track_bbox, conf in tracked_faces:
                        if Tracker._iou(bbox, track_bbox) > 0.5:
                            detections_map[track_id] = face
                            break

                detections_by_frame = detections_map
                
                if SHOW_DETECTION_TIME:
                    print(f"Detection: {time.time() - start:.3f}s")

            # ==========================
            # RECOGNITION (MANAGER DECIDES WHEN)
            # ==========================

            # Pass tracker output + detection map to manager
            recognition_results = recognition_manager.update(
                tracked_faces,
                detections_by_frame
            )

            # ==========================
            # DRAWING
            # ==========================

            for track_id, bbox, name, score in recognition_results:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Track ID + Name + Score
                label = f"ID:{track_id} {name} ({score:.2f})"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )
                
                # Registration
                if track_id in detections_by_frame:
                    registrar.process(detections_by_frame[track_id])

            # ==========================
            # FPS + STATUS
            # ==========================

            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            draw_registration_status(frame, registrar)
            
            # Recognition stats
            stats = recognition_manager.get_stats()
            stats_text = f"Tracks: {stats['total_tracks']} | Known: {stats['known_faces']}"
            cv2.putText(frame, stats_text, (20, CAMERA_HEIGHT - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.putText(
                frame,
                "R = Register | Q = Quit",
                (20, CAMERA_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            # ==========================
            # DISPLAY
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