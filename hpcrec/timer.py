import time


class Timer:
    def __init__(self, task: str):
        """
        Context manager for timing a block of code.
        """
        self.task: str = task

    def __enter__(self):
        print(f"STARTING {self.task}")
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t_start
        print(f"DONE with {self.task} in {t:.4f} seconds")
