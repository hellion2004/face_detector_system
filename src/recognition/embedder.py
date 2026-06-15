import numpy as np


class FaceEmbedder:

    def get_embedding(self, face):

        embedding = face.embedding.astype(np.float32)

        embedding /= np.linalg.norm(embedding)

        return embedding