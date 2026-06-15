from recognition.matcher import FaceMatcher


# ==========================
# SETTINGS
# ==========================

RECOGNIZE_EVERY_N_FRAMES = 15

# ==========================


class Recognizer:

    def __init__(
        self,
        database,
        embedder
    ):

        self.database = database

        self.embedder = embedder

        self.matcher = FaceMatcher()

    def recognize(
        self,
        face
    ):

        embedding = self.embedder.get_embedding(
            face
        )

        users = self.database.get_all_users()

        name, score = self.matcher.match(
            embedding,
            users
        )

        return name, score