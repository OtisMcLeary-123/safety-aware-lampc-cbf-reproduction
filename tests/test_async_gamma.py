from dataclasses import dataclass

import pytest

from lampc_cbf.async_gamma import (
    AsyncGammaWorker,
    AtomicGammaStore,
    GammaUpdate,
    GammaUpdateQueue,
)


def _update(gamma, version, created=10.0, deadline=20.0):
    return GammaUpdate(gamma, version, created, deadline, "feedback")


def test_store_applies_newest_valid_update_at_step_boundary():
    queue = GammaUpdateQueue()
    queue.publish(_update(0.08, 1))
    queue.publish(_update(0.02, 2))
    store = AtomicGammaStore(0.15)
    assert store.snapshot() == (0.15, 0)
    audit = store.apply_pending(queue, now=12.0)
    assert store.snapshot() == (0.02, 2)
    assert audit.applied.version == 2
    assert audit.rejected_old_version == 1


def test_store_rejects_expired_future_and_replayed_updates():
    queue = GammaUpdateQueue()
    queue.publish(_update(0.02, 1, created=1.0, deadline=2.0))
    queue.publish(_update(0.05, 2, created=20.0, deadline=30.0))
    store = AtomicGammaStore(0.15, clock_skew_tolerance=0.0)
    audit = store.apply_pending(queue, now=10.0)
    assert store.snapshot() == (0.15, 0)
    assert audit.rejected_expired == 1
    assert audit.rejected_future == 1

    queue.publish(_update(0.08, 3, created=9.0, deadline=11.0))
    store.apply_pending(queue, now=10.0)
    queue.publish(_update(0.02, 3, created=9.5, deadline=11.0))
    replay = store.apply_pending(queue, now=10.0)
    assert replay.rejected_old_version == 1
    assert store.snapshot() == (0.08, 3)


def test_bounded_queue_drops_oldest_without_blocking():
    queue = GammaUpdateQueue(maxsize=2)
    queue.publish(_update(0.15, 1))
    queue.publish(_update(0.11, 2))
    queue.publish(_update(0.08, 3))
    assert [item.version for item in queue.drain()] == [2, 3]
    assert queue.dropped_overflow == 1


@dataclass
class _Decision:
    gamma: float = 0.02
    provider: str = "local"
    model: str = "fake"


class _Mapper:
    def infer_gamma(self, instruction, *, current_gamma, feedback):
        assert instruction == "more clearance"
        assert current_gamma == 0.15
        assert feedback is True
        return _Decision()


def test_async_worker_publishes_versioned_update():
    queue = GammaUpdateQueue()
    with AsyncGammaWorker(_Mapper(), queue, clock=lambda: 100.0) as worker:
        request, future = worker.submit(
            "more clearance", current_gamma=0.15, ttl_seconds=2.0
        )
        update = future.result(timeout=1.0)
    assert request.version == 1
    assert update.valid_until == 102.0
    store = AtomicGammaStore(0.15)
    audit = store.apply_pending(queue, now=100.5)
    assert audit.applied.gamma == 0.02


class _BrokenMapper:
    def infer_gamma(self, *args, **kwargs):
        raise RuntimeError("network unavailable")


def test_async_worker_failure_never_changes_controller_parameter():
    queue = GammaUpdateQueue()
    with AsyncGammaWorker(_BrokenMapper(), queue, clock=lambda: 1.0) as worker:
        _, future = worker.submit("feedback", current_gamma=0.08, ttl_seconds=1.0)
        assert future.result(timeout=1.0) is None
    store = AtomicGammaStore(0.08)
    assert store.apply_pending(queue, now=1.5).applied is None
    assert store.snapshot() == (0.08, 0)


@pytest.mark.parametrize("gamma", [0.0, -0.1, 0.151, float("nan")])
def test_update_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        _update(gamma, 1)
