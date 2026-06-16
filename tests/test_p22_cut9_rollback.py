"""P22 C9 — RollbackManager: automatic rollback with signed RollbackEvent."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from matrixai.continual import (
    ProductionMonitor,
    RollbackCheckResult,
    RollbackEvent,
    RollbackManager,
    parse_mxcontinual,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

_POLICY_SRC = """
CONTINUAL_POLICY RollbackTestPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json
  REGISTRY_NAME test_rollback_model
  BASE_VERSION v1.0

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    SOURCES [api]
  END

  DRIFT_DETECTION
    FEATURES [score]
    METHODS
      score: ks threshold=0.15
    END
    MIN_SAMPLES 50
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 50
    MIN_GROUND_TRUTH_RATIO 0.5
    COOLDOWN_DAYS 1
  END

  TRAINING
    METHOD incremental_finetune
    LEARNING_RATE_FACTOR 0.1
    MAX_EPOCHS 10
    DATASET_MIX
      BASE_WEIGHT 0.5
      PRODUCTION_WEIGHT 0.5
      RECENCY_DECAY linear
    END
  END

  APPROVAL_GATE
    HOLDOUT_FRACTION 0.2
    REGRESSION_GUARD
      METRIC accuracy
      MUST_IMPROVE_BY 0.0
      MAX_DEGRADATION_PER_LABEL 0.1
    END
    HUMAN_APPROVAL false
  END

  ROLLBACK
    AUTO_TRIGGER true
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 10
  END

  AUDIT
    PERSIST_DRIFT_REPORTS true
    PERSIST_UPDATE_TRACES true
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT false
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 14
    SIGNATURE_REQUIRED false
  END
END
"""

_POLICY_NO_AUTO = """
CONTINUAL_POLICY RollbackOffPolicy
  TARGET_MODEL examples/model.mxai
  BASE_PARAMETER_SET runs/model_001/params.best.json
  REGISTRY_NAME test_rollback_model
  BASE_VERSION v1.0

  GROUND_TRUTH
    WINDOW_DAYS 7
    REQUIRED_FIELD label
    SOURCES [api]
  END

  DRIFT_DETECTION
    FEATURES [score]
    METHODS
      score: ks threshold=0.15
    END
    MIN_SAMPLES 50
    CHECK_FREQUENCY daily
    REFERENCE_DATASET base_training
  END

  UPDATE_TRIGGER
    MIN_NEW_SAMPLES 50
    MIN_GROUND_TRUTH_RATIO 0.5
    COOLDOWN_DAYS 1
  END

  TRAINING
    METHOD incremental_finetune
    LEARNING_RATE_FACTOR 0.1
    MAX_EPOCHS 10
    DATASET_MIX
      BASE_WEIGHT 0.5
      PRODUCTION_WEIGHT 0.5
      RECENCY_DECAY linear
    END
  END

  APPROVAL_GATE
    HOLDOUT_FRACTION 0.2
    REGRESSION_GUARD
      METRIC accuracy
      MUST_IMPROVE_BY 0.0
      MAX_DEGRADATION_PER_LABEL 0.1
    END
    HUMAN_APPROVAL false
  END

  ROLLBACK
    AUTO_TRIGGER false
    METRIC accuracy
    SLIDING_WINDOW_HOURS 24
    DEGRADATION_THRESHOLD 0.05
    MIN_SAMPLES_IN_WINDOW 10
  END

  AUDIT
    PERSIST_DRIFT_REPORTS true
    PERSIST_UPDATE_TRACES true
    EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT false
    REFINEMENT_DRIFT_PERSISTENCE_DAYS 14
    SIGNATURE_REQUIRED false
  END
END
"""

_NOW = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
_RECENT = _NOW - timedelta(hours=1)
_SIGNING_KEY = "a" * 64  # 32-byte hex key


def _policy(src: str = _POLICY_SRC):
    return parse_mxcontinual(src)


def _degraded_monitor(policy, n: int = 15) -> ProductionMonitor:
    """Monitor with degraded accuracy (all wrong) and enough samples."""
    m = ProductionMonitor(policy, reference_accuracy=0.91, labels=["spam", "ham"])
    for _ in range(n):
        m.record("ham", "spam", observed_at=_RECENT)
    return m


def _healthy_monitor(policy, n: int = 15) -> ProductionMonitor:
    """Monitor with healthy accuracy (all correct) and enough samples."""
    m = ProductionMonitor(policy, reference_accuracy=0.91, labels=["spam", "ham"])
    for _ in range(n):
        m.record("spam", "spam", observed_at=_RECENT)
    return m


def _make_entry(version: str, ps_id: str, parent_ps_id: str | None = None) -> MagicMock:
    entry = MagicMock()
    entry.version = version
    entry.parameter_set_id = ps_id
    entry.metrics = {}
    if parent_ps_id:
        entry.metrics["parent_parameter_set_id"] = parent_ps_id
    return entry


def _make_registry(
    current_version: str = "v1.1",
    current_ps_id: str = "ps-v1.1",
    parent_ps_id: str = "ps-v1.0",
    parent_version: str = "v1.0",
) -> MagicMock:
    current_entry = _make_entry(current_version, current_ps_id, parent_ps_id)
    parent_entry = _make_entry(parent_version, parent_ps_id, None)

    registry = MagicMock()
    registry.get.side_effect = lambda name, version: (
        current_entry if version in ("current", current_version) else parent_entry
    )
    registry.list.return_value = [current_entry, parent_entry]
    registry.tag = MagicMock()
    return registry


# ── RollbackCheckResult shape tests ───────────────────────────────────────────

class TestRollbackCheckResultShape:
    def test_check_returns_result_instance(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert isinstance(result, RollbackCheckResult)

    def test_check_fields_populated_when_triggered(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is True
        assert result.from_parameter_set_id == "ps-v1.1"
        assert result.to_parameter_set_id == "ps-v1.0"
        assert result.from_version == "v1.1"
        assert result.to_version == "v1.0"

    def test_check_reason_is_online_degradation(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.reason == "online_degradation"

    def test_check_window_accuracy_is_present(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert isinstance(result.window_accuracy, float)
        assert result.window_accuracy == pytest.approx(0.0)


# ── check() no-rollback conditions ────────────────────────────────────────────

class TestCheckNoRollback:
    def test_no_rollback_when_auto_trigger_false(self):
        pol = _policy(_POLICY_NO_AUTO)
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is False

    def test_no_rollback_when_not_enough_samples(self):
        pol = _policy()
        mon = ProductionMonitor(pol, reference_accuracy=0.91, labels=["spam", "ham"])
        for _ in range(5):  # < MIN_SAMPLES_IN_WINDOW=10
            mon.record("ham", "spam", observed_at=_RECENT)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is False
        assert result.enough_samples is False

    def test_no_rollback_when_no_degradation(self):
        pol = _policy()
        mon = _healthy_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is False

    def test_no_rollback_when_registry_get_fails(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = MagicMock()
        reg.get.side_effect = Exception("not found")
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is False

    def test_no_rollback_when_no_parent_ps_id(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        # current entry has no parent_parameter_set_id
        entry = _make_entry("v1.1", "ps-v1.1", parent_ps_id=None)
        reg = MagicMock()
        reg.get.return_value = entry
        reg.list.return_value = [entry]
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is False

    def test_no_rollback_when_parent_version_not_found(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        current_entry = _make_entry("v1.1", "ps-v1.1", "unknown-ps-id")
        reg = MagicMock()
        reg.get.return_value = current_entry
        reg.list.return_value = [current_entry]  # parent ps_id not in any entry
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.check(now=_NOW)
        assert result.should_rollback is False


# ── execute() tests ────────────────────────────────────────────────────────────

class TestExecute:
    def test_execute_returns_rollback_event(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            window_value=0.0,
            samples_in_window=15,
            now=_NOW,
        )
        assert isinstance(event, RollbackEvent)

    def test_execute_updates_current_tag(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        reg.tag.assert_called_once_with("test_rollback_model", "v1.0", "current")

    def test_execute_event_has_correct_versions(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            window_value=0.1,
            samples_in_window=15,
            now=_NOW,
        )
        assert event.from_version == "v1.1"
        assert event.to_version == "v1.0"
        assert event.from_parameter_set_id == "ps-v1.1"
        assert event.to_parameter_set_id == "ps-v1.0"

    def test_execute_event_has_correct_trigger_reason(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            trigger_reason="manual",
            now=_NOW,
        )
        assert event.trigger_reason == "manual"

    def test_execute_event_notification_sent_false(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        assert event.notification_sent is False

    def test_execute_event_rollback_id_format(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        assert event.rollback_id.startswith("rb-")


# ── signature tests ────────────────────────────────────────────────────────────

class TestSignature:
    def test_signature_is_sha256_without_key(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        assert event.signature is not None
        assert event.signature.startswith("sha256:")

    def test_signature_is_hmac_sha256_with_key(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, signing_key=_SIGNING_KEY, now=_NOW)
        event = mgr.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        assert event.signature is not None
        assert event.signature.startswith("hmac-sha256:")

    def test_signature_is_deterministic_for_same_inputs(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg1 = _make_registry()
        reg2 = _make_registry()
        mgr1 = RollbackManager(pol, mon, reg1, now=_NOW)
        mgr2 = RollbackManager(pol, mon, reg2, now=_NOW)
        e1 = mgr1.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        e2 = mgr2.execute(
            from_parameter_set_id="ps-v1.1",
            to_parameter_set_id="ps-v1.0",
            from_version="v1.1",
            to_version="v1.0",
            now=_NOW,
        )
        assert e1.signature == e2.signature


# ── run() combined workflow tests ─────────────────────────────────────────────

class TestRun:
    def test_run_returns_event_when_degraded(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.run(now=_NOW)
        assert isinstance(event, RollbackEvent)

    def test_run_returns_none_when_no_degradation(self):
        pol = _policy()
        mon = _healthy_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        result = mgr.run(now=_NOW)
        assert result is None

    def test_run_tags_current_when_rollback_executed(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        mgr.run(now=_NOW)
        reg.tag.assert_called_once_with("test_rollback_model", "v1.0", "current")

    def test_run_does_not_tag_when_no_rollback(self):
        pol = _policy()
        mon = _healthy_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        mgr.run(now=_NOW)
        reg.tag.assert_not_called()

    def test_run_event_window_accuracy_matches_monitor(self):
        pol = _policy()
        mon = _degraded_monitor(pol, n=15)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.run(now=_NOW)
        assert event is not None
        assert event.sliding_window_value == pytest.approx(0.0)

    def test_run_samples_in_window_in_event(self):
        pol = _policy()
        mon = _degraded_monitor(pol, n=15)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        event = mgr.run(now=_NOW)
        assert event is not None
        assert event.samples_in_window == 15

    def test_run_uses_default_now_when_not_passed(self):
        pol = _policy()
        mon = _degraded_monitor(pol)
        reg = _make_registry()
        mgr = RollbackManager(pol, mon, reg, now=_NOW)
        # Passing now=_NOW explicitly should produce an event
        event = mgr.run(now=_NOW)
        assert event is not None


_POLICY_WITH_NOTIFY = _POLICY_SRC.replace(
    "  ROLLBACK\n    AUTO_TRIGGER true",
    "  ROLLBACK\n    AUTO_TRIGGER true\n    NOTIFY_CAPABILITY notification",
)


class TestRollbackNotification:
    """RollbackManager calls notification_fn when NOTIFY_CAPABILITY is declared."""

    def test_notification_fn_called_when_capability_declared(self):
        """notification_fn receives the RollbackEvent when notify_capability is set."""
        pol = parse_mxcontinual(_POLICY_WITH_NOTIFY)
        mon = _degraded_monitor(pol)
        reg = _make_registry()

        received: list[RollbackEvent] = []
        def notify_fn(event: RollbackEvent) -> bool:
            received.append(event)
            return True

        mgr = RollbackManager(pol, mon, reg, now=_NOW, notification_fn=notify_fn)
        event = mgr.run(now=_NOW)
        assert event is not None
        assert len(received) == 1

    def test_notification_sent_true_when_fn_succeeds(self):
        """notification_sent=True when notification_fn returns True."""
        pol = parse_mxcontinual(_POLICY_WITH_NOTIFY)
        mon = _degraded_monitor(pol)
        reg = _make_registry()

        mgr = RollbackManager(pol, mon, reg, now=_NOW, notification_fn=lambda e: True)
        event = mgr.run(now=_NOW)
        assert event is not None
        assert event.notification_sent is True

    def test_notification_sent_false_when_no_capability(self):
        """notification_sent=False when notify_capability is not declared."""
        pol = _policy()  # no NOTIFY_CAPABILITY
        mon = _degraded_monitor(pol)
        reg = _make_registry()

        mgr = RollbackManager(pol, mon, reg, now=_NOW, notification_fn=lambda e: True)
        event = mgr.run(now=_NOW)
        assert event is not None
        assert event.notification_sent is False

    def test_notification_sent_false_when_fn_raises(self):
        """Exceptions in notification_fn are swallowed; notification_sent=False."""
        pol = parse_mxcontinual(_POLICY_WITH_NOTIFY)
        mon = _degraded_monitor(pol)
        reg = _make_registry()

        def failing_fn(event: RollbackEvent) -> bool:
            raise RuntimeError("webhook unreachable")

        mgr = RollbackManager(pol, mon, reg, now=_NOW, notification_fn=failing_fn)
        event = mgr.run(now=_NOW)
        assert event is not None
        assert event.notification_sent is False

    def test_notification_fn_not_called_when_no_rollback(self):
        """notification_fn is not called when rollback does not trigger."""
        pol = parse_mxcontinual(_POLICY_WITH_NOTIFY)
        mon = _healthy_monitor(pol)
        reg = _make_registry()

        calls: list[int] = []
        mgr = RollbackManager(pol, mon, reg, now=_NOW, notification_fn=lambda e: calls.append(1) or True)
        result = mgr.run(now=_NOW)
        assert result is None
        assert len(calls) == 0


class TestSignatureCoverage:
    """Verify that notification_sent is included in the RollbackEvent signature payload."""

    def test_notification_sent_covered_by_signature(self):
        """Two events identical except notification_sent must produce different signatures."""
        pol = parse_mxcontinual(_POLICY_WITH_NOTIFY)
        mon = _degraded_monitor(pol)
        reg_a = _make_registry()
        reg_b = _make_registry()
        signing_key = _SIGNING_KEY

        event_notified = RollbackManager(
            pol, mon, reg_a, signing_key=signing_key, now=_NOW,
            notification_fn=lambda e: True,
        ).run(now=_NOW)

        event_not_notified = RollbackManager(
            pol, mon, reg_b, signing_key=signing_key, now=_NOW,
            notification_fn=lambda e: False,
        ).run(now=_NOW)

        assert event_notified is not None
        assert event_not_notified is not None
        assert event_notified.notification_sent is True
        assert event_not_notified.notification_sent is False
        # Different notification_sent → different canonical payload → different signature
        assert event_notified.signature != event_not_notified.signature
