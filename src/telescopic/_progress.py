"""Progress of runs and tuning: a tqdm bar when tqdm is installed, lines on stderr otherwise."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Seconds between the lines printed without tqdm.
INTERVAL = 10.0

# Whether the tuning functions show their steps. The sampler sets it around Kernel.tune,
# whose signature, which custom kernels implement, carries no progress flag.
_TUNING = ContextVar("tuning_progress", default=False)


class Progress:
    # Units (replicates, or steps of tuning) done out of a total.
    def __init__(self, total: int, description: str, unit: str = "replicate") -> None:
        self.total, self.description, self.unit, self.finished = total, description, unit, 0
        self.start = self.shown = time.perf_counter()
        try:
            from tqdm import tqdm
        except ImportError:
            self.bar = None
        else:
            self.bar = tqdm(total=total, desc=description, unit=unit)

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
            f"telescopic: {self.description}: {self.finished} of {self.total} {self.unit}s "
            f"finished in {self.shown - self.start:.0f} s",
            file=sys.stderr,
        )


@contextmanager
def tuning(show: bool) -> Iterator[None]:
    # Within it, the tuning functions show their steps if show is true.
    token = _TUNING.set(show)
    try:
        yield
    finally:
        _TUNING.reset(token)


def tuning_steps(steps: range, description: str) -> Iterator[int]:
    # The steps of a tuning loop, shown as they finish within tuning(True).
    if not _TUNING.get():
        yield from steps
        return
    bar = Progress(len(steps), description, "step")
    try:
        for done, step in enumerate(steps, 1):
            yield step
            bar.update(done)
    finally:
        bar.close()
