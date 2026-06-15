import time


# ==========================
# SETTINGS
# ==========================

NUM_SAMPLES = 15

MIN_DETECTION_SCORE = 0.7

SAMPLE_INTERVAL_SECONDS = 0.5

# ==========================


class Registrar:

    def __init__(
        self,
        database,
        embedder
    ):

        self.database = database

        self.embedder = embedder

        self.active = False

        self.name = ""

        self.samples_collected = 0

        self.last_sample_time = 0

    def start(self):

        self.name = input(
            "\nEnter name: "
        ).strip()

        if not self.name:
            return

        self.active = True

        self.samples_collected = 0

        self.last_sample_time = 0

        print(
            f"\nRegistering {self.name}"
        )

    def process(
        self,
        face
    ):

        if not self.active:
            return

        if face.det_score < MIN_DETECTION_SCORE:
            return

        current_time = time.time()

        if (
            current_time -
            self.last_sample_time
            <
            SAMPLE_INTERVAL_SECONDS
        ):
            return

        self.last_sample_time = current_time

        embedding = self.embedder.get_embedding(
            face
        )

        self.database.add_embedding(
            self.name,
            embedding
        )

        self.samples_collected += 1

        print(
            f"Sample "
            f"{self.samples_collected}/"
            f"{NUM_SAMPLES}"
        )

        if self.samples_collected >= NUM_SAMPLES:

            self.active = False

            print(
                f"\nSaved {self.name}"
            )