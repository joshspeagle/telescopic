"""Progress of runs: a tqdm bar when tqdm is installed, lines on stderr otherwise."""

from __future__ import annotations

import sys
import time

# Seconds between the lines printed without tqdm.
INTERVAL = 10.0


class Progress:
    # Replicates finished out of a run's total.
    def __init__(self, total: int, description: str) -> None:
        self.total, self.description, self.finished = total, description, 0
        self.start = self.shown = time.perf_counter()
        try:
            from tqdm.auto import tqdm
        except ImportError:
            self.bar = None
        else:
            self.bar = tqdm(total=total, desc=description, unit="replicate")

    def update(self, finished: int) -> None:
        if finished == self.finished:
            return
        if self.bar is not None:
            self.bar.update(finished - self.finished)
        self.finished = finished
        if self.bar is None and time.perf_counter() - self.shown >= INTERVAL:
            self._line()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        else:
            self._line()

    def _line(self) -> None:
        self.shown = time.perf_counter()
        print(
            f"telescopic: {self.description}: {self.finished} of {self.total} replicates "
            f"finished in {self.shown - self.start:.0f} s",
            file=sys.stderr,
        )
