"""Console progress reporting for experiment-domain runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, Protocol

from repro_core.context.events import EpochEvent, TrainEndEvent, UpdateEvent

ProgressMode = Literal["auto", "none", "line", "tqdm"]


class ProgressReporter(Protocol):
    def set_total_updates(self, total: int, *, completed: int = 0) -> None: ...
    def advance_to(self, update: int, *, epoch: int = 0) -> None: ...
    def on_update(self, event: UpdateEvent) -> None: ...
    def on_epoch(self, event: EpochEvent) -> None: ...
    def on_train_end(self, event: TrainEndEvent) -> None: ...
    def write(self, message: str) -> None: ...
    def close(self) -> None: ...


class NullProgressReporter:
    def set_total_updates(self, total: int, *, completed: int = 0) -> None:
        return None

    def advance_to(self, update: int, *, epoch: int = 0) -> None:
        return None

    def on_update(self, event: UpdateEvent) -> None:
        return None

    def on_epoch(self, event: EpochEvent) -> None:
        return None

    def on_train_end(self, event: TrainEndEvent) -> None:
        return None

    def write(self, message: str) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class RunProgressContext:
    label: str
    index: int
    count: int
    total_updates: int | None = None


class LineProgressReporter:
    def __init__(self, context: RunProgressContext, *, every: int = 1) -> None:
        self.context = context
        self.every = max(1, every)

    def set_total_updates(self, total: int, *, completed: int = 0) -> None:
        return None

    def advance_to(self, update: int, *, epoch: int = 0) -> None:
        if update % self.every == 0:
            self.write(
                f"[{self.context.index}/{self.context.count}] {self.context.label} "
                f"update={update} epoch={epoch}"
            )

    def on_update(self, event: UpdateEvent) -> None:
        self.advance_to(event.update, epoch=event.epoch)

    def on_epoch(self, event: EpochEvent) -> None:
        self.write(
            f"[{self.context.index}/{self.context.count}] {self.context.label} "
            f"epoch={event.epoch} updates={event.end_update}"
        )

    def on_train_end(self, event: TrainEndEvent) -> None:
        self.write(
            f"[{self.context.index}/{self.context.count}] {self.context.label} "
            f"done reason={event.reason} updates={event.update} epoch={event.epoch}"
        )

    def write(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def close(self) -> None:
        return None


class TqdmProgressReporter:
    def __init__(
        self, context: RunProgressContext, *, every: int = 1, position: int = 1
    ) -> None:
        from tqdm import tqdm

        self.context = context
        self.every = max(1, every)
        self._last_update = 0
        self._bar = tqdm(
            total=context.total_updates,
            desc=context.label,
            unit="upd",
            dynamic_ncols=True,
            leave=False,
            position=position,
            file=sys.stderr,
        )

    def set_total_updates(self, total: int, *, completed: int = 0) -> None:
        self._bar.total = total
        self._bar.n = completed
        self._last_update = completed
        self._bar.refresh()

    def advance_to(self, update: int, *, epoch: int = 0) -> None:
        delta = max(0, update - self._last_update)
        if delta:
            self._bar.update(delta)
            self._last_update = update
        if update % self.every == 0:
            self._bar.set_postfix({"epoch": epoch, "update": update}, refresh=True)

    def on_update(self, event: UpdateEvent) -> None:
        self.advance_to(event.update, epoch=event.epoch)

    def on_epoch(self, event: EpochEvent) -> None:
        self._bar.set_postfix(
            {"epoch": event.epoch, "update": event.end_update}, refresh=True
        )

    def on_train_end(self, event: TrainEndEvent) -> None:
        self._bar.set_postfix(
            {"reason": event.reason, "update": event.update}, refresh=True
        )

    def write(self, message: str) -> None:
        from tqdm import tqdm

        tqdm.write(message, file=sys.stderr)

    def close(self) -> None:
        self._bar.close()


class ProgressManager:
    def __init__(self, *, mode: ProgressMode, every: int, total_runs: int) -> None:
        self.requested_mode = mode
        self.mode: Literal["none", "line", "tqdm"] = _resolve_mode(mode)
        self.every = max(1, every)
        self.total_runs = total_runs
        self._outer = None
        self._completed = 0
        if self.mode == "tqdm":
            from tqdm import tqdm

            self._outer = tqdm(
                total=total_runs,
                desc="runs",
                unit="run",
                dynamic_ncols=True,
                leave=True,
                position=0,
                file=sys.stderr,
            )

    def reporter(self, context: RunProgressContext) -> ProgressReporter:
        if self.mode == "none":
            return NullProgressReporter()
        if self.mode == "line":
            return LineProgressReporter(context, every=self.every)
        return TqdmProgressReporter(context, every=self.every)

    def on_run_start(self, context: RunProgressContext) -> None:
        if self.mode == "line":
            print(
                f"[{context.index}/{context.count}] start {context.label}",
                file=sys.stderr,
                flush=True,
            )
        elif self._outer is not None:
            self._outer.set_postfix_str(context.label, refresh=True)

    def on_run_end(self) -> None:
        self._completed += 1
        if self._outer is not None:
            self._outer.update(1)

    def write(self, message: str) -> None:
        if self.mode == "tqdm":
            from tqdm import tqdm

            tqdm.write(message, file=sys.stderr)
        elif self.mode == "line":
            print(message, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._outer is not None:
            self._outer.close()


def _resolve_mode(mode: ProgressMode) -> Literal["none", "line", "tqdm"]:
    if mode == "auto":
        return "tqdm" if sys.stderr.isatty() else "line"
    return mode
