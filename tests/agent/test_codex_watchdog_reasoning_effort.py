"""Regression tests for #112909: watchdogs must scale with reasoning effort.

Salvages the effort-scaling mechanism from the unmerged #41442 (authorship
preserved on the source commit).

The bug: Hermes's three non-stream watchdogs (no-event TTFB, event-idle,
stale-call) pick their fuses from the estimated prompt token count alone. A
small prompt gets the shortest fuses (12s idle / 120s TTFB / 90s stale), but
GPT-5-family models at ``reasoning_effort=high|xhigh`` routinely think
silently for 100-170s before their first byte. The watchdogs then kill a
healthy, hard-thinking model and retry into the same wall.

The contract under test (behavioral, via the real resolver
``_resolve_nonstream_watchdogs`` — no source reading, no value snapshots):

- high/xhigh effort scales all three timeouts by the multiplier (default 5x);
- low/medium/none/minimal/absent effort leaves every resolved value
  BIT-IDENTICAL to the no-effort baseline (zero behavior change is the hard
  contract for non-high tiers);
- the env override knobs (HERMES_CODEX_HIGH_EFFORT_MULTIPLIER,
  HERMES_API_CALL_STALE_REASONING_MULTIPLIER, and the pre-existing explicit
  HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS) still work.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

_EFFORT_ENV_VARS = (
    "HERMES_CODEX_HIGH_EFFORT_MULTIPLIER",
    "HERMES_API_CALL_STALE_REASONING_MULTIPLIER",
    "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS",
    "HERMES_CODEX_TTFB_TIMEOUT_SECONDS",
)


def _make_codex_agent(tmp_path, monkeypatch):
    for var in _EFFORT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def _small_payload(effort=None, *, chat_completions_shape=False):
    """A tiny prompt (<10K est. tokens): the shortest-fuse tier."""
    payload = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}
    if effort is not None:
        if chat_completions_shape:
            payload["reasoning_effort"] = effort
        else:
            payload["reasoning"] = {"effort": effort}
    return payload


def _watchdogs(agent, payload):
    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs

    return _resolve_nonstream_watchdogs(agent, payload)


@pytest.mark.parametrize("effort", ["high", "xhigh"])
@pytest.mark.parametrize("chat_completions_shape", [False, True])
def test_high_effort_scales_all_three_watchdogs(
    tmp_path, monkeypatch, effort, chat_completions_shape
):
    """high/xhigh on a small prompt multiplies TTFB, idle, and stale by 5x."""
    agent = _make_codex_agent(tmp_path, monkeypatch)
    baseline = _watchdogs(agent, _small_payload())
    scaled = _watchdogs(
        agent, _small_payload(effort, chat_completions_shape=chat_completions_shape)
    )
    assert scaled.stale_timeout == pytest.approx(baseline.stale_timeout * 5.0)
    assert scaled.ttfb_timeout == pytest.approx(baseline.ttfb_timeout * 5.0)
    assert scaled.idle_timeout == pytest.approx(baseline.idle_timeout * 5.0)


@pytest.mark.parametrize("effort", ["low", "medium", "none", "minimal", ""])
def test_non_high_effort_is_bit_identical_to_no_effort(tmp_path, monkeypatch, effort):
    """Hard contract: non-high tiers change NOTHING — every resolved field,
    for both payload shapes (and a malformed reasoning field), equals the
    no-effort baseline exactly."""
    agent = _make_codex_agent(tmp_path, monkeypatch)
    baseline = _watchdogs(agent, _small_payload())
    for payload in (
        _small_payload(effort),
        _small_payload(effort, chat_completions_shape=True),
        {**_small_payload(), "reasoning": effort},  # malformed: not a dict
    ):
        assert _watchdogs(agent, payload) == baseline


def test_effort_env_knobs_still_work(tmp_path, monkeypatch):
    """The multiplier env vars tune the scaling, an explicit idle timeout
    still overrides the (scaled) tier default, and the agent-level
    reasoning_config fallback scales when the payload carries no effort."""
    agent = _make_codex_agent(tmp_path, monkeypatch)
    baseline = _watchdogs(agent, _small_payload())

    monkeypatch.setenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", "2.0")
    monkeypatch.setenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", "3.0")
    scaled = _watchdogs(agent, _small_payload("high"))
    assert scaled.ttfb_timeout == pytest.approx(baseline.ttfb_timeout * 2.0)
    assert scaled.idle_timeout == pytest.approx(baseline.idle_timeout * 2.0)
    assert scaled.stale_timeout == pytest.approx(baseline.stale_timeout * 3.0)

    # Explicit operator-set idle timeout beats the scaled default outright.
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "77")
    explicit = _watchdogs(agent, _small_payload("high"))
    assert explicit.idle_timeout == 77.0
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")

    # Payload wins over the agent attr; the attr is only a fallback.
    agent.reasoning_config = {"effort": "low"}
    assert _watchdogs(agent, _small_payload("high")).idle_timeout == pytest.approx(
        baseline.idle_timeout * 2.0
    )
    agent.reasoning_config = {"effort": "high"}
    assert _watchdogs(agent, _small_payload()).idle_timeout == pytest.approx(
        baseline.idle_timeout * 2.0
    )
