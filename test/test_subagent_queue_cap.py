"""Tests for the flood-scoped subagent spawn queue cap.

The :class:`SubagentManager` docstring promises that a "max concurrent
limit prevents resource exhaustion." The promise was half-kept: when
the running pool is saturated, new spawns are *queued* rather than
started, so ``_running_count`` stays bounded — but ``_queue`` itself
had no upper bound, and a sustained non-batched flood would grow it
linearly with the spawn rate while the drain stays bounded.

The cap is **flood-scoped, not wave-scoped**. ``spawn_run tasks=[...]``
advertises "still pass ALL of them in one call — any beyond the cap are
queued and drained automatically", and a wave's worst case is already
bounded by ``batch_total`` (transport-clamped at 1000). Refusing a wave
member mid-fan-out would strand the digest, because the spawn discipline
forbids the parent from retrying mid-wave. So the cap fires only when
no ``batch_id`` was supplied — the genuine flood shape (a tight
``spawn_run`` loop without a wave identity, or a retry storm from a
transport that misses releases). Batched spawns skip the cap entirely.

A related leak lives in ``_announce_rejection``: the older
``self._tasks[f"reject-{info.id}"] = asyncio.ensure_future(...)``
pattern left every prefixed key in place forever — no pop matched any
of the four prefixed insertion sites (``reject-`` ×2, ``lost-``,
``flush-``). The fix routes batched announces through the module-level
``_safe_fire`` helper, which uses ``_background_tasks`` with its own
discard callback. The same helper closes the leak for every sibling
refusal that goes through ``_announce_rejection``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kiro_crew.subagent import SubagentManager

#: The two host-memory guards ``SubagentManager.spawn`` consults
#: (``check_memory_available`` against ``agent.spawn_min_memory_gb`` and
#: ``cached_admission_check``'s posture tier) read the operator's machine
#: rather than the test's own input, and a refusal surfaces as a bare
#: ``KeyError`` one line later. Pin them so a memory-pressured CI runner
#: does not produce that misleading failure. Inner patches in each test
#: land on top of this and vary the verdict to whatever that test needs.
pytestmark = pytest.mark.usefixtures("healthy_host_memory")


def _admitted():
    """A trivial admitted admission decision; posture is irrelevant to the queue cap."""
    from kiro_crew import resource_status as rs

    return rs.AdmissionDecision(admitted=True, posture=rs.POSTURE_AMPLE, available_gb=16.0)


def _mgr(max_concurrent: int = 1, on_done=None):
    """Build a SubagentManager with ``max_concurrent=1`` so every spawn is queued."""
    return SubagentManager(
        sessions=MagicMock(),
        ctx_builder=MagicMock(),
        on_done=on_done if on_done is not None else MagicMock(),
        max_concurrent=max_concurrent,
    )


class TestSubagentQueueCap:
    """Bounded queue depth for non-batched subagent spawns under sustained flood."""

    def test_spawn_is_queued_when_pool_is_saturated(self) -> None:
        """A non-batched flood's first member lands queued, not running.

        With ``max_concurrent=1`` and no agents currently running, the pool is
        already saturated for the purpose of the stagger check (every spawn
        goes through ``_should_stagger_queue`` and sees ``slot_free=True`` only
        for the very first call). We force the second member into the queue by
        pinning ``_running_count`` to the cap before the second call.
        """
        mgr = _mgr(max_concurrent=1)
        with (
            patch("kiro_crew.subagent.check_memory_available", return_value=(True, 8.0)),
            patch("kiro_crew.subagent.KiroCrewConfig") as mock_cfg,
            patch("kiro_crew.subagent.cached_admission_check", return_value=_admitted()),
            patch("kiro_crew.subagent.validate_cwd", return_value=("", None)),
            patch("kiro_crew.subagent.sel") as mock_sel,
            patch("kiro_crew.subagent._validate_agent", return_value=("", None)),
        ):
            mock_cfg.load.return_value.agent.spawn_min_memory_gb = 4.0
            mock_sel.return_value.log_tool_invocation = MagicMock()

            # First call: pool is empty, spawn starts. To prove the second call
            # really does queue (not bypass), pin _running_count after the first.
            info1 = mgr.spawn(task="first task", parent_session_key="sess-1")
            assert info1 is not None
            assert not info1.queued
            mgr._running_count = 1  # pin to cap

            info2 = mgr.spawn(task="second task", parent_session_key="sess-1")
            assert info2 is not None
            assert info2.queued is True
            assert len(mgr._queue) == 1

    def test_non_batched_flood_past_queue_cap_is_refused(self) -> None:
        """A non-batched flood past the queue cap is refused with a typed SEL outcome.

        With ``max_concurrent=1`` and the default ``_max_queued=32``, a tight
        non-batched spawn loop that fills the queue and overflows it must be
        refused with a typed ``refused_queue_full`` SEL outcome and a retry-later
        error string, mirroring the existing ``refused_memory_critical`` shape.
        """
        mgr = _mgr(max_concurrent=1)
        mgr._max_queued = 2  # pin the cap low for the test
        with (
            patch("kiro_crew.subagent.check_memory_available", return_value=(True, 8.0)),
            patch("kiro_crew.subagent.KiroCrewConfig") as mock_cfg,
            patch("kiro_crew.subagent.cached_admission_check", return_value=_admitted()),
            patch("kiro_crew.subagent.validate_cwd", return_value=("", None)),
            patch("kiro_crew.subagent.sel") as mock_sel,
            patch("kiro_crew.subagent._validate_agent", return_value=("", None)),
        ):
            mock_cfg.load.return_value.agent.spawn_min_memory_gb = 4.0
            mock_sel.return_value.log_tool_invocation = MagicMock()

            # Pin the pool to the cap so every spawn goes to the queue.
            mgr._running_count = 1

            # First two non-batched queue entries fit under the cap.
            q1 = mgr.spawn(task="queued task 1", parent_session_key="sess-1")
            assert q1 is not None and q1.queued is True
            q2 = mgr.spawn(task="queued task 2", parent_session_key="sess-1")
            assert q2 is not None and q2.queued is True
            assert len(mgr._queue) == 2

            # The third non-batched call overflows the cap and must be refused.
            info = mgr.spawn(task="overflow task", parent_session_key="sess-1")
            assert info is not None
            assert info.done is True
            assert info.queued is False
            assert info.batch_id == ""
            assert "queue" in info.error.lower()
            assert "retry" in info.error.lower()

            # Typed SEL outcome mirroring the existing pattern.
            typed_call = [
                c
                for c in mock_sel.return_value.log_tool_invocation.call_args_list
                if c.kwargs.get("outcome") == "refused_queue_full"
            ]
            assert len(typed_call) == 1
            assert typed_call[0].kwargs["tool_name"] == "spawn_run"
            assert typed_call[0].kwargs["metadata"]["max_queued"] == 2
            assert typed_call[0].kwargs["metadata"]["queue_depth"] == 2

            # The queue must NOT have grown past the cap.
            assert len(mgr._queue) == 2

    def test_batched_wave_member_bypasses_queue_cap(self) -> None:
        """A wave member skips the queue cap entirely.

        ``spawn_run tasks=[...]`` advertises "still pass ALL of them in one
        call — any beyond the cap are queued and drained automatically".
        Refusing a wave member mid-fan-out would strand the digest because
        the same spawn discipline forbids the parent from retrying mid-wave.
        So a batched call (``batch_id`` set) bypasses the cap: every member
        of a single spawn_run wave is queued regardless of depth, with
        ``batch_total`` already bounding the wave's worst case.
        """
        mgr = _mgr(max_concurrent=1)
        mgr._max_queued = 1  # pin the cap low to make the bypass visible
        with (
            patch("kiro_crew.subagent.check_memory_available", return_value=(True, 8.0)),
            patch("kiro_crew.subagent.KiroCrewConfig") as mock_cfg,
            patch("kiro_crew.subagent.cached_admission_check", return_value=_admitted()),
            patch("kiro_crew.subagent.validate_cwd", return_value=("", None)),
            patch("kiro_crew.subagent.sel") as mock_sel,
            patch("kiro_crew.subagent._validate_agent", return_value=("", None)),
        ):
            mock_cfg.load.return_value.agent.spawn_min_memory_gb = 4.0
            mock_sel.return_value.log_tool_invocation = MagicMock()
            mgr._running_count = 1  # pin the pool so every spawn goes to the queue

            # Fill the queue to the cap with batched members — every one lands queued,
            # none refused, because batched calls bypass the cap.
            for i in range(10):
                info = mgr.spawn(
                    task=f"wave member {i}",
                    parent_session_key="sess-1",
                    batch_id="BIG-WAVE",
                    batch_total=10,
                )
                assert info is not None, f"wave member {i} returned None"
                assert info.queued is True, f"wave member {i} was refused: {info.error}"
                assert info.batch_id == "BIG-WAVE"

            # Queue grew past the cap because the cap does not apply to batched calls.
            assert len(mgr._queue) == 10

            # A non-batched call arriving at the same queue depth is the one refused.
            info = mgr.spawn(task="non-batched flood", parent_session_key="sess-2")
            assert info is not None
            assert info.done is True
            assert info.batch_id == ""


class TestAnnounceRejectionSelfReap:
    """``_announce_rejection`` must not grow ``_tasks`` for batched rejections.

    The pre-existing helper inserted ``f"reject-{info.id}"`` into ``_tasks``
    for every batched rejection so the wave accounting could release the
    digest. No pop ever matched the prefix — ``_tasks.pop(...)`` calls
    unprefixed ``agent_id`` keys only. A sustained flood of rejections
    would grow that map without bound. The new path routes the on_done
    callback through the module-level ``_safe_fire`` helper, which uses
    ``_background_tasks`` (a set cleaned via
    ``add_done_callback(_background_tasks.discard)``) — the canonical
    fire-and-forget shape. This test pins that contract for the most
    reachable batched rejection shape (governance denial).
    """

    def test_batched_governance_rejection_does_not_grow_tasks(self) -> None:
        """A batched governance denial announces via _safe_fire, not _tasks."""
        from kiro_crew.subagent import _vet_spawn_governance

        on_done = MagicMock()
        mgr = _mgr(max_concurrent=1, on_done=on_done)
        tasks_before = len(mgr._tasks)

        with (
            patch("kiro_crew.subagent.check_memory_available", return_value=(True, 8.0)),
            patch("kiro_crew.subagent.KiroCrewConfig") as mock_cfg,
            patch("kiro_crew.subagent.cached_admission_check", return_value=_admitted()),
            patch("kiro_crew.subagent.validate_cwd", return_value=("", None)),
            patch("kiro_crew.subagent.sel") as mock_sel,
            patch("kiro_crew.subagent._validate_agent", return_value=("", None)),
            patch("kiro_crew.subagent._vet_spawn_governance", return_value="test denied"),
        ):
            mock_cfg.load.return_value.agent.spawn_min_memory_gb = 4.0
            mock_sel.return_value.log_tool_invocation = MagicMock()

            info = mgr.spawn(
                task="batched governance test",
                parent_session_key="sess-1",
                batch_id="BG1",
                batch_total=2,
            )
            assert info is not None
            assert info.done is True
            assert info.batch_id == "BG1"
            assert "denied" in info.error.lower()

        # No reject-<id> entry landed in _tasks.
        assert len(mgr._tasks) == tasks_before, (
            f"batched rejection must not grow _tasks "
            f"(was {tasks_before}, now {len(mgr._tasks)})"
        )
        assert all(
            not k.startswith("reject-") for k in mgr._tasks
        ), f"_tasks contains a reject-* entry: {list(mgr._tasks)}"
        # Silence the unused-import warning for _vet_spawn_governance — the
        # import is a guard that the helper is bound at module scope, not a
        # value this test calls directly.
        del _vet_spawn_governance
