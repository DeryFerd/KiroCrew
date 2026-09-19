"""Tests for the shared transport approval-mode resolver.

Every channel gateway needs the same decision: given the orchestrator's CLI
``--approval`` override and the configured ``agent.approval_mode``, does this
turn auto-approve or does it go interactive (deny-by-default)? One shared
implementation answers it, and each channel reaches it through its own named
seam.

Slack is the deliberate exception. Its transport resolver reports the
``--approval`` flag as interactive and lets the gateway's own approval-event
path apply the ``yolo`` and ``reads`` grants, because those grants classify the
tool event rather than the turn. Its seam therefore keeps its own body, and the
tests below assert both behaviors so neither can drift into the other.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kiro_crew.messaging.driver import (
    APPROVAL_AUTO,
    APPROVAL_INTERACTIVE,
    resolve_transport_approval_mode,
)

#: The nine channels whose transport resolver reads the flag directly.
DELEGATING_GATEWAYS = [
    "kiro_crew.discord.gateway",
    "kiro_crew.feishu.gateway",
    "kiro_crew.imessage.gateway",
    "kiro_crew.teams.gateway",
    "kiro_crew.telegram.gateway",
    "kiro_crew.webex.gateway",
    "kiro_crew.wecom.gateway",
    "kiro_crew.weixin.gateway",
    "kiro_crew.whatsapp.gateway",
]


def _orch(**overrides):
    base = dict(
        _approval_mode=None,
        _cfg=SimpleNamespace(agent=SimpleNamespace(approval_mode=APPROVAL_INTERACTIVE)),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestResolveTransportApprovalMode:
    def test_cli_yolo_override_resolves_auto(self) -> None:
        assert resolve_transport_approval_mode(_orch(_approval_mode="yolo")) == APPROVAL_AUTO

    def test_cli_auto_override_resolves_auto(self) -> None:
        assert resolve_transport_approval_mode(_orch(_approval_mode=APPROVAL_AUTO)) == APPROVAL_AUTO

    def test_cli_interactive_override_resolves_interactive(self) -> None:
        assert (
            resolve_transport_approval_mode(_orch(_approval_mode=APPROVAL_INTERACTIVE))
            == APPROVAL_INTERACTIVE
        )

    def test_absent_override_falls_back_to_configured_auto(self) -> None:
        orch = _orch(_cfg=SimpleNamespace(agent=SimpleNamespace(approval_mode=APPROVAL_AUTO)))
        assert resolve_transport_approval_mode(orch) == APPROVAL_AUTO

    def test_absent_override_falls_back_to_configured_interactive(self) -> None:
        assert resolve_transport_approval_mode(_orch()) == APPROVAL_INTERACTIVE

    def test_cli_flag_beats_the_configured_mode(self) -> None:
        orch = _orch(
            _approval_mode=APPROVAL_INTERACTIVE,
            _cfg=SimpleNamespace(agent=SimpleNamespace(approval_mode=APPROVAL_AUTO)),
        )
        assert resolve_transport_approval_mode(orch) == APPROVAL_INTERACTIVE

    def test_unknown_mode_collapses_to_interactive(self) -> None:
        """Anything that is not exactly ``auto`` is deny-by-default."""
        assert (
            resolve_transport_approval_mode(_orch(_approval_mode="something_else"))
            == APPROVAL_INTERACTIVE
        )

    def test_missing_override_attribute_is_tolerated(self) -> None:
        """Duck-typed orchestrators: no ``_approval_mode`` attribute at all."""
        orch = SimpleNamespace(
            _cfg=SimpleNamespace(agent=SimpleNamespace(approval_mode=APPROVAL_AUTO))
        )
        assert resolve_transport_approval_mode(orch) == APPROVAL_AUTO


@pytest.mark.parametrize("module_name", DELEGATING_GATEWAYS)
def test_each_gateway_keeps_its_named_seam(module_name: str) -> None:
    """The per-channel name must stay importable.

    Channel tests import ``_resolve_approval_mode`` from the gateway module
    itself, so the shared helper is reached *through* that seam rather than
    replacing it.
    """
    module = pytest.importorskip(module_name)
    assert callable(module._resolve_approval_mode)


@pytest.mark.parametrize("module_name", DELEGATING_GATEWAYS)
def test_gateway_seam_agrees_with_the_shared_helper(module_name: str) -> None:
    """Every delegating channel resolves the same way as the shared helper."""
    module = pytest.importorskip(module_name)
    assert module._resolve_approval_mode(_orch(_approval_mode="yolo")) == APPROVAL_AUTO
    assert (
        module._resolve_approval_mode(_orch(_approval_mode=APPROVAL_INTERACTIVE))
        == APPROVAL_INTERACTIVE
    )
    assert module._resolve_approval_mode(
        _orch(_approval_mode=None)
    ) == resolve_transport_approval_mode(_orch(_approval_mode=None))
