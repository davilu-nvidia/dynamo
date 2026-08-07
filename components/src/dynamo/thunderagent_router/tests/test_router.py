# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ThunderAgentScheduler that don't need a Dynamo runtime."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.router import ThunderAgentConfig, ThunderAgentScheduler

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@dataclass
class FakeCapacity:
    """Stand-in for WorkerCapacityProvider that returns a fixed snapshot."""

    workers: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[int, int]:
        return dict(self.workers)


def make_router(
    capacity_workers: Optional[dict[int, int]] = None,
    config: Optional[ThunderAgentConfig] = None,
) -> tuple[ThunderAgentScheduler, FakeCapacity]:
    capacity = FakeCapacity(workers=capacity_workers or {})
    cfg = config or ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
        pause_threshold=0.95,
        soft_demote_threshold=0.80,
    )
    return ThunderAgentScheduler(capacity, cfg), capacity  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_first_turn_no_admission_block():
    router, _ = make_router()
    decision = await router.before_request("p1")
    assert decision.was_paused is False
    assert decision.priority_jump == 0.0


@pytest.mark.asyncio
async def test_after_request_records_real_tokens():
    router, _ = make_router()
    await router.before_request("p1")
    await router.after_request("p1", prompt_tokens=120, completion_tokens=30)
    program = router._table.programs["p1"]
    assert program.token_total == 150
    assert program.status == ProgramStatus.ACTING


@pytest.mark.asyncio
async def test_status_snapshot_reports_programs_and_worker_utilization():
    workers = {
        1: 1000,
        2: 500,
    }
    router, _ = make_router(capacity_workers=workers)

    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=25)
    await router.before_request("p2", estimated_prompt_tokens=50)

    snapshot = await router.status_snapshot()

    assert snapshot["programs_total"] == 2
    assert snapshot["paused_total"] == 0
    assert snapshot["lifecycle_counts"]["active"] == 2
    assert snapshot["status_counts"]["acting"] == 1
    assert snapshot["status_counts"]["reasoning"] == 1
    assert snapshot["workers"]["1"]["capacity"] == 1000
    assert snapshot["workers"]["1"]["used"] == 225
    assert snapshot["workers"]["1"]["active_programs"] == 1
    assert {
        (program["program_id"], program["assigned_worker_id"])
        for program in snapshot["programs"]
    } == {("p1", 1), ("p2", 2)}


@pytest.mark.asyncio
async def test_metrics_snapshot_reports_lifecycle_counters_and_gauges():
    router, _ = make_router(capacity_workers={1: 1000})

    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=20)
    assert await router.end_program("p1") is True

    async def fail_status_snapshot() -> dict:
        raise AssertionError("metrics_snapshot must not build detailed status rows")

    router.status_snapshot = fail_status_snapshot  # type: ignore[method-assign]

    metrics = await router.metrics_snapshot()

    assert metrics["counters"]["programs_created_total"] == 1
    assert metrics["counters"]["programs_ended_total"] == 1
    assert metrics["counters"]["requests_admitted_total"] == 1
    assert metrics["counters"]["worker_assignments_total"] == 1
    assert metrics["gauges"]["programs_total"] == 0
    assert metrics["gauges"]["paused_total"] == 0
    assert metrics["gauges"]["workers_total"] == 1


@pytest.mark.asyncio
async def test_before_request_records_exact_prompt_estimate_before_admission():
    router, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=1234)
    program = router._table.programs["p1"]
    assert program.token_total == 1234
    assert program.status == ProgramStatus.REASONING


@pytest.mark.asyncio
async def test_assigned_worker_hint_reflects_sticky_assignment():
    router, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.assign_worker("p1", 3)
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 3


@pytest.mark.asyncio
async def test_pause_acting_then_before_request_blocks_until_resume():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
    )
    router, _ = make_router(config=cfg)

    await router.before_request("p1")
    await router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.PAUSED

    waiter = asyncio.create_task(router.before_request("p1"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)

    async with router._lock:
        router._resume_program(router._table.programs["p1"], target_worker_id=1)

    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True
    assert decision.priority_jump == cfg.resume_priority_boost
    assert decision.assigned_worker_hint == 1
    metrics = await router.metrics_snapshot()
    assert metrics["counters"]["worker_assignments_total"] == 2


@pytest.mark.asyncio
async def test_forced_resume_after_timeout():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=0.05,
    )
    router, _ = make_router(config=cfg)
    await router.before_request("p1")
    await router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    decision = await router.before_request("p1")
    assert decision.was_paused is True
    assert router._stat_forced_resumes >= 1
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_new_program_queues_before_first_request_when_capacity_full():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=2.0,
        pause_threshold=1.0,
        resume_hysteresis=0.0,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("existing", estimated_prompt_tokens=950)
    await router.assign_worker("existing", 1)

    waiter = asyncio.create_task(
        router.before_request("new", estimated_prompt_tokens=100)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
    assert router._table.programs["new"].lifecycle == ProgramLifecycle.PAUSED

    async with router._lock:
        router._resume_program(router._table.programs["new"], target_worker_id=1)
    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True


@pytest.mark.asyncio
async def test_cold_start_admits_without_sticky_pin():
    """No MDC visible yet: don't park, let the request through; the
    chunk-loop callback will populate ``assigned_worker_id`` once the
    engine picks a worker."""
    router, _ = make_router(capacity_workers={})
    decision = await router.before_request("cold_start")
    assert decision.was_paused is False
    assert decision.assigned_worker_hint is None
    program = router._table.programs["cold_start"]
    assert program.lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_soft_demote_marks_borderline_workers():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        soft_demote_threshold=0.80,
        pause_threshold=0.95,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("p1")
    await router.assign_worker("p1", 1)
    await router.after_request("p1", prompt_tokens=750, completion_tokens=0)
    await router.before_request("p1")
    await router.assign_worker("p1", 1)

    router._apply_soft_demotes(router._capacity.snapshot())
    program = router._table.programs["p1"]
    assert program.soft_demoted_until > time.monotonic()

    await router.after_request("p1", prompt_tokens=860, completion_tokens=2)
    decision = await router.before_request("p1")
    assert decision.priority_jump == cfg.soft_demote_priority_jump
    assert decision.was_soft_demoted is True


@pytest.mark.asyncio
async def test_pause_until_safe_pauses_smallest_acting_first():
    cfg = ThunderAgentConfig(
        pause_threshold=0.80,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)

    # Used = 600 + 100 + 2*100 = 900; pausing small leaves 700 <= target.
    for pid, prompt_tokens in [("big", 600), ("small", 100)]:
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_until_safe_is_scoped_to_overloaded_worker():
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
        2: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)

    for pid, worker_id, prompt_tokens in [
        ("hot_big", 1, 700),
        ("hot_small", 1, 200),
        ("cold", 2, 700),
    ]:
        await router.before_request(pid)
        await router.assign_worker(pid, worker_id)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["hot_small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["hot_big"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["cold"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_drives_util_to_pause_target_not_threshold():
    """Each pause cycle drains util down to pause_target, not just below threshold."""
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1_000_000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100_000, completion_tokens=0)

    await router._pause_until_safe(router._capacity.snapshot())

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    # 10 programs * (100k tokens + 100 buffer) = 1.0010M; target 0.80M.
    # Each pause releases (100k + 100). Pause 2 -> 0.8008M (still over),
    # pause 3 -> 0.7007M (under). Anything else means over- or under-shoot.
    assert paused == 3, f"paused={paused}"


@pytest.mark.asyncio
async def test_scheduler_tick_resumes_before_pausing_new_overload():
    """Upstream TA ordering: resume old paused work, then pause overload."""
    cfg = ThunderAgentConfig(
        pause_threshold=1.0,
        pause_target=0.80,
        resume_hysteresis=0.0,
        acting_token_weight=1.0,
        acting_decay_tau_seconds=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
    }
    router, capacity = make_router(config=cfg)

    # Capacity is attached after setup so first-turn admission gating does not
    # queue the synthetic programs before the scheduler tick.
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100, completion_tokens=0)
        router._table.programs[pid].acting_since = time.monotonic() - 10.0

    capacity.workers = workers
    await router._scheduler_tick()

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    assert paused == 6


# ---------------------------------------------------------------------------
# Gap-harvest warmup
# ---------------------------------------------------------------------------


def warmup_config(**overrides) -> ThunderAgentConfig:
    defaults = dict(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
        warmup_enabled=True,
        warmup_util_threshold=0.50,
        warmup_lead_seconds=5.0,
        warmup_min_samples=2,
        warmup_max_per_tick=4,
    )
    defaults.update(overrides)
    return ThunderAgentConfig(**defaults)


async def run_one_step(router: ThunderAgentScheduler, program_id: str) -> None:
    await router.before_request(program_id)
    await router.after_request(program_id, prompt_tokens=100, completion_tokens=50)


@pytest.mark.asyncio
async def test_gap_samples_recorded_on_return():
    router, _ = make_router(capacity_workers={1: 10_000})
    await run_one_step(router, "p1")
    program = router._table.programs["p1"]
    # Backdate the acting gap, then return for the next step.
    program.acting_since = time.monotonic() - 3.0
    await run_one_step(router, "p1")
    assert len(program.gap_samples) == 1
    assert program.gap_samples[0] == pytest.approx(3.0, abs=0.5)


@pytest.mark.asyncio
async def test_warmup_fires_once_per_gap():
    fired: list[tuple[str, int]] = []

    async def warmup_fn(program_id: str, worker_id: int) -> bool:
        fired.append((program_id, worker_id))
        return True

    router, capacity = make_router(
        capacity_workers={1: 10_000}, config=warmup_config()
    )
    router.set_warmup_callback(warmup_fn)
    await run_one_step(router, "p1")

    program = router._table.programs["p1"]
    program.gap_samples = [8.0, 8.0]
    program.acting_since = time.monotonic() - 4.0  # elapsed >= 8 - 5

    await router._maybe_warmup(capacity.snapshot())
    await asyncio.sleep(0.05)
    assert fired == [("p1", 1)]
    assert program.warmup_step == program.step_count

    # Same gap: no second warmup.
    await router._maybe_warmup(capacity.snapshot())
    await asyncio.sleep(0.05)
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_warmup_waits_for_predicted_return():
    fired: list[str] = []

    async def warmup_fn(program_id: str, worker_id: int) -> bool:
        fired.append(program_id)
        return True

    router, capacity = make_router(
        capacity_workers={1: 10_000}, config=warmup_config()
    )
    router.set_warmup_callback(warmup_fn)
    await run_one_step(router, "p1")

    program = router._table.programs["p1"]
    program.gap_samples = [60.0, 60.0]
    program.acting_since = time.monotonic() - 1.0  # elapsed < 60 - 5

    await router._maybe_warmup(capacity.snapshot())
    await asyncio.sleep(0.05)
    assert fired == []


@pytest.mark.asyncio
async def test_warmup_gated_by_utilization():
    fired: list[str] = []

    async def warmup_fn(program_id: str, worker_id: int) -> bool:
        fired.append(program_id)
        return True

    # token_total 150 + buffer 100 = 250 used; capacity 400 -> util 0.625.
    router, capacity = make_router(
        capacity_workers={1: 400}, config=warmup_config()
    )
    router.set_warmup_callback(warmup_fn)
    await run_one_step(router, "p1")

    program = router._table.programs["p1"]
    program.gap_samples = [1.0, 1.0]
    program.acting_since = time.monotonic() - 10.0

    await router._maybe_warmup(capacity.snapshot())
    await asyncio.sleep(0.05)
    assert fired == []
    assert router._stat_warmups_skipped_util == 1


@pytest.mark.asyncio
async def test_warmup_disabled_by_default():
    fired: list[str] = []

    async def warmup_fn(program_id: str, worker_id: int) -> bool:
        fired.append(program_id)
        return True

    router, capacity = make_router(capacity_workers={1: 10_000})
    router.set_warmup_callback(warmup_fn)
    await run_one_step(router, "p1")

    program = router._table.programs["p1"]
    program.gap_samples = [1.0, 1.0]
    program.acting_since = time.monotonic() - 10.0

    await router._maybe_warmup(capacity.snapshot())
    await asyncio.sleep(0.05)
    assert fired == []


# ---------------------------------------------------------------------------
# Prefill admission pacing
# ---------------------------------------------------------------------------

from dynamo.thunderagent_router.router import PrefillPacer  # noqa: E402


@pytest.mark.asyncio
async def test_pacer_limits_inflight_and_releases_sjf():
    pacer = PrefillPacer(limit=1, max_wait_seconds=5.0)
    await pacer.acquire(1, cost=100)  # takes the slot

    order: list[str] = []

    async def contender(name: str, cost: float):
        await pacer.acquire(1, cost)
        order.append(name)

    big = asyncio.create_task(contender("big", 5000))
    await asyncio.sleep(0.01)
    small = asyncio.create_task(contender("small", 10))
    await asyncio.sleep(0.01)
    assert order == []  # both queued behind the held slot

    pacer.release(1)  # slot transfers to cheapest waiter first
    await asyncio.sleep(0.01)
    assert order == ["small"]
    pacer.release(1)
    await asyncio.sleep(0.01)
    assert order == ["small", "big"]
    await asyncio.gather(big, small)
    assert pacer.stat_paced_total == 2
    assert pacer.stat_timeouts_total == 0


@pytest.mark.asyncio
async def test_pacer_timeout_lets_request_proceed():
    pacer = PrefillPacer(limit=1, max_wait_seconds=0.05)
    await pacer.acquire(1, cost=1)
    await pacer.acquire(1, cost=1)  # times out, proceeds anyway
    assert pacer.stat_timeouts_total == 1
    # Both slots eventually released without underflow.
    pacer.release(1)
    pacer.release(1)
    pacer.release(1)  # extra release is a no-op
    await pacer.acquire(1, cost=1)  # slot available again immediately


@pytest.mark.asyncio
async def test_pacer_workers_are_independent():
    pacer = PrefillPacer(limit=1, max_wait_seconds=5.0)
    await pacer.acquire(1, cost=1)
    await pacer.acquire(2, cost=1)  # different worker: no queueing
    assert pacer.stat_paced_total == 0


@pytest.mark.asyncio
async def test_pacing_cost_uses_last_completion_then_prompt():
    router, _ = make_router(capacity_workers={1: 10_000})
    d1 = await router.before_request("p1", estimated_prompt_tokens=1200)
    assert d1.pacing_cost == 1200  # first step: whole prompt is cold
    await router.after_request("p1", prompt_tokens=1200, completion_tokens=333)
    d2 = await router.before_request("p1", estimated_prompt_tokens=1600)
    assert d2.pacing_cost == 333  # later steps: last completion (divergence)


@pytest.mark.asyncio
async def test_first_step_exempt_from_pacing():
    router, _ = make_router(capacity_workers={1: 10_000})
    d1 = await router.before_request("p1", estimated_prompt_tokens=1200)
    assert d1.pace_eligible is False  # cold first step: never paced
    await router.after_request("p1", prompt_tokens=1200, completion_tokens=50)
    d2 = await router.before_request("p1", estimated_prompt_tokens=1300)
    assert d2.pace_eligible is True
