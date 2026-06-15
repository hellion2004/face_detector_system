import time


class FPSCounter:

    def __init__(self):

        self.prev_time = time.time()

    def update(self):

        current = time.time()

        fps = 1 / (current - self.prev_time)

        self.prev_time = current

        return int(fps)