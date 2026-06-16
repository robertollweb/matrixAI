"""P20 C1 — Parser e IR para .mxact y POLICY real_with_audit en .mxai."""
import pytest

from matrixai.actions import (
    ActionContractSpec,
    MxactParseError,
    parse_mxact,
    CAPABILITIES,
    HIGH_RISK_CAPABILITIES,
    MUTATING_CAPABILITIES,
)
from matrixai.parser.parser import MatrixAIParseError, parse_text


# ── fixture sources ────────────────────────────────────────────────────────────

_EMAIL_MXACT = """
ACTION_CONTRACT SendNotification
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com", "alerts@example.com"]
    allowed_domains    = ["example.com"]
    max_subject_length = 200
    max_body_length    = 5000
  END
  DRY_RUN required
  ROLLBACK send_correction_email
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=5 per_hour=30
  SIGNATURE_REQUIRED true
END

ROLLBACK send_correction_email
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com", "alerts@example.com"]
    template = "correction"
  END
END
"""

_HTTP_POST_MXACT = """
ACTION_CONTRACT CreateTicket
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://api.example.com/tickets"]
    required_headers = ["Authorization", "Content-Type"]
    max_payload_bytes = 10240
    timeout_seconds = 5
  END
  DRY_RUN required
  ROLLBACK delete_ticket
  SANDBOX not_required
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=10 per_hour=100
  SIGNATURE_REQUIRED true
END

ROLLBACK delete_ticket
  CAPABILITY http_post
  SCOPE
    allowed_urls = ["https://api.example.com/tickets/delete"]
    timeout_seconds = 5
  END
END
"""

_FILESYSTEM_WRITE_MXACT = """
ACTION_CONTRACT WriteAuditLog
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["/var/log/matrixai/audit/"]
    max_file_size_bytes = 1048576
    allowed_extensions = [".log", ".json"]
  END
  DRY_RUN required
  ROLLBACK delete_audit_log
  SANDBOX required
  SANDBOX_LIMITS
    max_memory_mb = 64
    max_cpu_seconds = 5
    max_wall_seconds = 10
    no_network = true
    allowed_env_vars = ["MATRIXAI_AUDIT_DIR"]
  END
  HUMAN_APPROVAL false
  RATE_LIMIT per_minute=60 per_hour=3600
  SIGNATURE_REQUIRED true
END

ROLLBACK delete_audit_log
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["/var/log/matrixai/audit/"]
    operation = "delete"
  END
END
"""

_MXAI_REAL_ACTION = """
PROJECT AlertSystem

VECTOR Alert[3]
  risk_score: Probability
  severity: Score[0, 10]
  confidence: Confidence
END

NETWORK AlertClassifier
  INPUT Alert
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT alert_prob: Probability
END

ACTION SendAlert
  TARGET email_send
  POLICY real_with_audit
  CONDITION alert_prob > 0.9
  INPUT recipient: String, subject: String, body: String
END

GRAPH
  Alert -> AlertClassifier
  AlertClassifier -> SendAlert
END

AUDIT
  EXPLAIN Alert -> AlertClassifier -> SendAlert
END
"""

_MXAI_SIMULATE_DEFAULT = """
PROJECT Demo

VECTOR Input[2]
  x: Probability
  y: Probability
END

ACTION Respond
  WHEN x > 0.5
  CALL simulated.respond
END

GRAPH
  Input -> Respond
END
"""


# ── C1 tests ───────────────────────────────────────────────────────────────────

class TestMxactParserEmailSend:
    def test_mxact_parser_accepts_email_send_contract(self):
        contracts = parse_mxact(_EMAIL_MXACT)
        assert len(contracts) == 1
        c = contracts[0]
        assert isinstance(c, ActionContractSpec)
        assert c.name == "SendNotification"
        assert c.capability == "email_send"

    def test_email_contract_scope_parsed(self):
        c = parse_mxact(_EMAIL_MXACT)[0]
        assert c.scope["allowed_recipients"] == ["ops@example.com", "alerts@example.com"]
        assert c.scope["max_subject_length"] == 200

    def test_email_contract_rollback_resolved(self):
        c = parse_mxact(_EMAIL_MXACT)[0]
        assert c.rollback is not None
        assert c.rollback.name == "send_correction_email"
        assert c.rollback.capability == "email_send"
        assert c.rollback.scope["template"] == "correction"

    def test_email_contract_rate_limit(self):
        c = parse_mxact(_EMAIL_MXACT)[0]
        assert c.rate_limit is not None
        assert c.rate_limit.per_minute == 5
        assert c.rate_limit.per_hour == 30

    def test_email_contract_flags(self):
        c = parse_mxact(_EMAIL_MXACT)[0]
        assert c.dry_run_required is True
        assert c.sandbox_required is False
        assert c.human_approval is False
        assert c.signature_required is True


class TestMxactParserHttpPost:
    def test_mxact_parser_accepts_http_post_with_rollback(self):
        contracts = parse_mxact(_HTTP_POST_MXACT)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.capability == "http_post"
        assert c.rollback is not None
        assert c.rollback.name == "delete_ticket"

    def test_http_post_scope_url_list(self):
        c = parse_mxact(_HTTP_POST_MXACT)[0]
        assert c.scope["allowed_urls"] == ["https://api.example.com/tickets"]
        assert c.scope["timeout_seconds"] == 5


class TestMxactParserFilesystemWrite:
    def test_mxact_parser_accepts_filesystem_write_with_sandbox(self):
        contracts = parse_mxact(_FILESYSTEM_WRITE_MXACT)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.capability == "filesystem_write"
        assert c.sandbox_required is True

    def test_filesystem_sandbox_limits_parsed(self):
        c = parse_mxact(_FILESYSTEM_WRITE_MXACT)[0]
        lim = c.sandbox_limits
        assert lim is not None
        assert lim.max_memory_mb == 64
        assert lim.max_cpu_seconds == 5
        assert lim.max_wall_seconds == 10
        assert lim.no_network is True
        assert lim.allowed_env_vars == ["MATRIXAI_AUDIT_DIR"]

    def test_filesystem_rollback_scope(self):
        c = parse_mxact(_FILESYSTEM_WRITE_MXACT)[0]
        assert c.rollback is not None
        assert c.rollback.scope["operation"] == "delete"


class TestMxactParserRejections:
    def test_mxact_parser_rejects_unknown_capability(self):
        src = """
ACTION_CONTRACT Foo
  CAPABILITY fly_to_moon
  SCOPE
    target = "moon"
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED true
END
"""
        with pytest.raises(MxactParseError, match="unknown capability"):
            parse_mxact(src)

    def test_mxact_parser_rejects_high_risk_without_sandbox(self):
        src = """
ACTION_CONTRACT BadWrite
  CAPABILITY database_write
  SCOPE
    allowed_tables = ["logs"]
  END
  DRY_RUN required
  ROLLBACK undo_write
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED true
END

ROLLBACK undo_write
  CAPABILITY database_write
  SCOPE
    operation = "delete"
  END
END
"""
        with pytest.raises(MxactParseError, match="high-risk.*SANDBOX required"):
            parse_mxact(src)

    def test_mxact_parser_rejects_human_approval_without_channel(self):
        src = """
ACTION_CONTRACT ApproveMe
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["https://api.example.com/data"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL true
  SIGNATURE_REQUIRED true
END
"""
        with pytest.raises(MxactParseError, match="APPROVAL_CHANNEL"):
            parse_mxact(src)

    def test_mxact_parser_rejects_wildcard_only_scope(self):
        src = """
ACTION_CONTRACT WildGet
  CAPABILITY http_get
  SCOPE
    allowed_urls = ["*"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED true
END
"""
        with pytest.raises(MxactParseError, match="wildcard"):
            parse_mxact(src)

    def test_mxact_parser_rejects_wildcard_path_scope(self):
        src = """
ACTION_CONTRACT WildWrite
  CAPABILITY filesystem_write
  SCOPE
    allowed_paths = ["**"]
    allowed_extensions = [".log"]
  END
  DRY_RUN required
  ROLLBACK undo
  SANDBOX required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED true
END

ROLLBACK undo
  CAPABILITY filesystem_write
  SCOPE
    operation = "delete"
  END
END
"""
        with pytest.raises(MxactParseError, match="wildcard"):
            parse_mxact(src)

    def test_mxact_parser_rejects_missing_rollback_for_mutating_action(self):
        src = """
ACTION_CONTRACT NoRollback
  CAPABILITY email_send
  SCOPE
    allowed_recipients = ["ops@example.com"]
  END
  DRY_RUN required
  SANDBOX not_required
  HUMAN_APPROVAL false
  SIGNATURE_REQUIRED true
END
"""
        with pytest.raises(MxactParseError, match="requires ROLLBACK"):
            parse_mxact(src)


class TestMxaiParserP20Extensions:
    def test_mxai_parser_accepts_policy_real_with_audit(self):
        prog = parse_text(_MXAI_REAL_ACTION)
        action = prog.actions[0]
        assert action.policy == "real_with_audit"

    def test_mxai_parser_action_has_target(self):
        prog = parse_text(_MXAI_REAL_ACTION)
        action = prog.actions[0]
        assert action.target == "email_send"

    def test_mxai_parser_action_condition_from_condition_keyword(self):
        prog = parse_text(_MXAI_REAL_ACTION)
        action = prog.actions[0]
        assert "alert_prob" in action.when
        assert action.condition.source == "alert_prob"

    def test_mxai_parser_input_params_parsed(self):
        prog = parse_text(_MXAI_REAL_ACTION)
        action = prog.actions[0]
        assert len(action.input_params) == 3
        assert action.input_params[0].name == "recipient"
        assert action.input_params[0].type == "String"
        assert action.input_params[1].name == "subject"
        assert action.input_params[2].name == "body"

    def test_mxai_parser_keeps_simulate_only_as_default(self):
        prog = parse_text(_MXAI_SIMULATE_DEFAULT)
        action = prog.actions[0]
        assert action.policy == "simulate_only"
        assert action.target == ""
        assert action.input_params == ()

    def test_mxai_parser_simulate_only_call_preserved(self):
        prog = parse_text(_MXAI_SIMULATE_DEFAULT)
        action = prog.actions[0]
        assert action.call == "simulated.respond"


class TestCapabilityRegistry:
    def test_all_nine_capabilities_present(self):
        expected = {
            "http_get", "http_post", "notification", "email_send",
            "database_read", "database_write", "filesystem_read",
            "filesystem_write", "subprocess_spawn",
        }
        assert set(CAPABILITIES.keys()) == expected

    def test_high_risk_capabilities_classified(self):
        assert "database_write" in HIGH_RISK_CAPABILITIES
        assert "filesystem_write" in HIGH_RISK_CAPABILITIES
        assert "subprocess_spawn" in HIGH_RISK_CAPABILITIES
        assert "http_get" not in HIGH_RISK_CAPABILITIES

    def test_mutating_capabilities_require_rollback(self):
        for cap in ("http_post", "email_send", "database_write", "filesystem_write"):
            assert cap in MUTATING_CAPABILITIES
        assert "http_get" not in MUTATING_CAPABILITIES
        assert "database_read" not in MUTATING_CAPABILITIES
