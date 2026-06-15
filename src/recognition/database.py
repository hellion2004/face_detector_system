import json
import os


DB_PATH = "database/face_db.json"


class FaceDatabase:

    def __init__(self):

        self.initialize_db()

    def initialize_db(self):

        default_data = {
            "users": {}
        }

        if not os.path.exists(DB_PATH):

            with open(DB_PATH, "w") as f:

                json.dump(
                    default_data,
                    f,
                    indent=4
                )

            return

        try:

            with open(DB_PATH, "r") as f:

                json.load(f)

        except Exception:

            print(
                "Database corrupted or empty."
                " Recreating..."
            )

            with open(DB_PATH, "w") as f:

                json.dump(
                    default_data,
                    f,
                    indent=4
                )

    def load(self):

        with open(DB_PATH, "r") as f:

            return json.load(f)

    def save(self, data):

        with open(DB_PATH, "w") as f:

            json.dump(
                data,
                f,
                indent=4
            )

    def add_embedding(
        self,
        name,
        embedding
    ):

        data = self.load()

        if name not in data["users"]:

            data["users"][name] = {
                "embeddings": []
            }

        data["users"][name]["embeddings"].append(
            embedding.tolist()
        )

        self.save(data)

    def get_all_users(self):

        return self.load()["users"]