import numpy as np


# ==========================
# SETTINGS
# ==========================

MATCH_THRESHOLD = 0.55

# ==========================


class FaceMatcher:

    def cosine_similarity(
        self,
        emb1,
        emb2
    ):

        return np.dot(
            emb1,
            emb2
        )

    def match(
        self,
        embedding,
        users
    ):

        best_name = "Unknown"

        best_score = 0

        for name, user_data in users.items():

            embeddings = user_data[
                "embeddings"
            ]

            for stored_embedding in embeddings:

                stored_embedding = np.array(
                    stored_embedding,
                    dtype=np.float32
                )

                score = self.cosine_similarity(
                    embedding,
                    stored_embedding
                )

                if score > best_score:

                    best_score = score

                    best_name = name

        if best_score < MATCH_THRESHOLD:

            best_name = "Unknown"

        return best_name, best_score