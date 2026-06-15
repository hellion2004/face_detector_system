import time


class FPSCounter:

    def __init__(self):

        self.prev_time = time.time()
        self.fps = 0

    def update(self):

        current = time.time()

        instant_fps = 1 / max(current - self.prev_time, 0.0001)

        self.prev_time = current

        self.fps = (self.fps * 0.9) + (instant_fps * 0.1)

        return int(self.fps)