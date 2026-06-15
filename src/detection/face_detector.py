from insightface.app import FaceAnalysis


# ==========================
# TUNABLE PARAMETERS
# ==========================

MODEL_NAME = "buffalo_s"

DETECTION_SIZE = (320, 320)

# ==========================


class FaceDetector:

    def __init__(self):

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=DETECTION_SIZE
        )

    def detect(self, frame):
        return self.app.get(frame)