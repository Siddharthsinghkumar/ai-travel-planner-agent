import pytest
from core.structured_logging import _redact_pii, _PII_REDACTION_MASK

def test_pii_redaction_email():
    text = "My email is test@example.com, please contact me."
    redacted = _redact_pii(text)
    assert _PII_REDACTION_MASK in redacted
    assert "test@example.com" not in redacted

def test_pii_redaction_phone():
    text = "Call me at +1 (555) 123-4567 or 9876543210."
    redacted = _redact_pii(text)
    assert "555" not in redacted
    assert "9876543210" not in redacted

def test_pii_redaction_passport():
    text = "My passport is A12345678, valid until 2030."
    redacted = _redact_pii(text)
    assert "A12345678" not in redacted
    assert _PII_REDACTION_MASK in redacted
