import time


class Timer:
    def __init__(self, task: str, verbose: bool = True):
        """
        Context manager for timing a block of code.
        """
        self.task: str = task
        self.verbose: bool = verbose
        self.t_start: float = -1.0
        self.t_end: float = -2.0

    @property
    def elapsed(self) -> float:
        """
        Elapsed time in seconds.
        """
        return self.t_end - self.t_start

    def __enter__(self):
        if self.verbose:
            print(f"STARTING {self.task}")
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        if self.verbose:
            print(f"DONE with {self.task} in {self.elapsed:.4f} seconds")
