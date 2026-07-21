"""Asynchronous, versioned language-to-MPC gamma update channel.

The design mirrors LeRobot's separation between a slow inference producer and
a time-critical robot client: inference never blocks the control loop, updates
carry deadlines, and the active parameter changes atomically at a step boundary.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from queue import Empty, Full, Queue
from threading import Lock
from time import time
from typing import Any, Callable


def _valid_gamma(value: float, upper: float = 0.15) -> float:
    """Default upper bound is the paper-experiment interval; the
    paper-continuous mode (registry 3.3) may widen it to 1.0."""

    converted = float(value)
    if not isfinite(converted) or not 0.0 < converted <= upper:
        raise ValueError(f"gamma must be finite and in (0, {upper:g}]")
    return converted


@dataclass(frozen=True, slots=True)
class GammaUpdate:
    gamma: float
    version: int
    created_at: float
    valid_until: float
    reason: str
    source: str = "llm"

    def __post_init__(self) -> None:
        # Absolute published bound (0, 1]; the store enforces the tighter
        # configured interval at apply time.
        _valid_gamma(self.gamma, upper=1.0)
        if self.version < 1:
            raise ValueError("version must be positive")
        if not isfinite(self.created_at) or not isfinite(self.valid_until):
            raise ValueError("timestamps must be finite")
        if self.valid_until <= self.created_at:
            raise ValueError("valid_until must be later than created_at")
        if not self.reason.strip() or not self.source.strip():
            raise ValueError("reason and source must be non-empty")


@dataclass(frozen=True, slots=True)
class GammaApplyAudit:
    applied: GammaUpdate | None
    drained: int
    rejected_expired: int
    rejected_old_version: int
    rejected_future: int


class GammaUpdateQueue:
    """Bounded MPSC queue that favors the newest feedback under overload."""

    def __init__(self, maxsize: int = 16) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: Queue[GammaUpdate] = Queue(maxsize=maxsize)
        self.dropped_overflow = 0
        self._drop_lock = Lock()

    def publish(self, update: GammaUpdate) -> None:
        """Publish without blocking; evict the oldest item if the queue is full."""

        try:
            self._queue.put_nowait(update)
            return
        except Full:
            pass
        try:
            self._queue.get_nowait()
        except Empty:  # another consumer made room
            pass
        else:
            with self._drop_lock:
                self.dropped_overflow += 1
        try:
            self._queue.put_nowait(update)
        except Full:
            # A concurrent producer won the slot. Dropping this message is
            # preferable to blocking the inference callback or control loop.
            with self._drop_lock:
                self.dropped_overflow += 1

    def drain(self) -> tuple[GammaUpdate, ...]:
        items: list[GammaUpdate] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                return tuple(items)


class AtomicGammaStore:
    """Controller-owned gamma, updated only by :meth:`apply_pending`."""

    def __init__(
        self,
        initial_gamma: float,
        *,
        clock_skew_tolerance: float = 0.05,
        gamma_upper: float = 0.15,
    ) -> None:
        self.gamma_upper = float(gamma_upper)
        self._gamma = _valid_gamma(initial_gamma, upper=self.gamma_upper)
        self._version = 0
        self._lock = Lock()
        if clock_skew_tolerance < 0.0 or not isfinite(clock_skew_tolerance):
            raise ValueError("clock_skew_tolerance must be finite and non-negative")
        self.clock_skew_tolerance = clock_skew_tolerance

    def snapshot(self) -> tuple[float, int]:
        with self._lock:
            return self._gamma, self._version

    def apply_pending(
        self, updates: GammaUpdateQueue, *, now: float | None = None
    ) -> GammaApplyAudit:
        """Drain, validate, and atomically apply only the newest valid update."""

        timestamp = time() if now is None else float(now)
        if not isfinite(timestamp):
            raise ValueError("now must be finite")
        drained = updates.drain()
        expired = old = future = 0
        with self._lock:
            eligible: list[GammaUpdate] = []
            for update in drained:
                if update.version <= self._version:
                    old += 1
                elif update.created_at > timestamp + self.clock_skew_tolerance:
                    future += 1
                elif update.valid_until < timestamp:
                    expired += 1
                elif not 0.0 < update.gamma <= self.gamma_upper:
                    # Out of the configured range: rejected as stale-invalid,
                    # visible in the audit trail rather than silently applied.
                    old += 1
                else:
                    eligible.append(update)
            selected = max(eligible, key=lambda item: item.version, default=None)
            if selected is not None:
                self._gamma = selected.gamma
                self._version = selected.version
            # Valid but superseded messages are old from the controller's point
            # of view and remain visible in the audit trail.
            old += max(0, len(eligible) - int(selected is not None))
        return GammaApplyAudit(selected, len(drained), expired, old, future)


@dataclass(frozen=True, slots=True)
class GammaRequest:
    version: int
    instruction: str
    current_gamma: float
    created_at: float
    valid_until: float
    reason: str


class AsyncGammaWorker:
    """Single inference worker that publishes results to a GammaUpdateQueue."""

    def __init__(
        self,
        mapper: Any,
        updates: GammaUpdateQueue,
        *,
        clock: Callable[[], float] = time,
    ) -> None:
        self.mapper = mapper
        self.updates = updates
        self.clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gamma-llm")
        self._version = 0
        self._lock = Lock()

    def submit(
        self,
        instruction: str,
        *,
        current_gamma: float,
        ttl_seconds: float,
        reason: str = "online_feedback",
    ) -> tuple[GammaRequest, Future[GammaUpdate | None]]:
        if not instruction.strip() or not reason.strip():
            raise ValueError("instruction and reason must be non-empty")
        _valid_gamma(current_gamma, upper=1.0)
        if not isfinite(ttl_seconds) or ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be finite and positive")
        created = float(self.clock())
        with self._lock:
            self._version += 1
            version = self._version
        request = GammaRequest(
            version,
            instruction.strip(),
            current_gamma,
            created,
            created + ttl_seconds,
            reason.strip(),
        )
        return request, self._executor.submit(self._infer_and_publish, request)

    def _infer_and_publish(self, request: GammaRequest) -> GammaUpdate | None:
        try:
            decision = self.mapper.infer_gamma(
                request.instruction,
                current_gamma=request.current_gamma,
                feedback=True,
            )
            update = GammaUpdate(
                gamma=decision.gamma,
                version=request.version,
                created_at=request.created_at,
                valid_until=request.valid_until,
                reason=request.reason,
                source=f"{decision.provider}:{decision.model}",
            )
        except Exception:
            # No exception crosses into the real-time loop. The absence of an
            # update preserves the last validated, locally stored parameter.
            return None
        self.updates.publish(update)
        return update

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> AsyncGammaWorker:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
