import logging

from core.logging_config import SensitiveDataRedactionFilter, _redact_text, _redact_value


def test_redact_text_masks_query_and_bearer_tokens():
    text = "https://example.test/path?api_key=abcd1234&token=zzzz Authorization=Bearer super.secret.token"
    redacted = _redact_text(text)
    assert "api_key=***REDACTED***" in redacted
    assert "token=***REDACTED***" in redacted
    assert "Authorization=***REDACTED***" in redacted
    assert "super.secret.token" not in redacted


def test_redact_value_masks_sensitive_nested_payload_fields():
    payload = {
        "booking_request": {
            "url": "https://provider.example/checkout?token=abc123",
            "post_data": {
                "sessionToken": "token-value",
                "fare": "1000",
                "Authorization": "Bearer abcdef",
            },
        }
    }
    redacted = _redact_value(payload)
    assert redacted["booking_request"]["post_data"]["sessionToken"] == "***REDACTED***"
    assert redacted["booking_request"]["post_data"]["Authorization"] == "***REDACTED***"
    assert redacted["booking_request"]["post_data"]["fare"] == "1000"
    assert "token=***REDACTED***" in redacted["booking_request"]["url"]


def test_redaction_filter_masks_msg_args_and_sensitive_extra_attrs():
    record = logging.LogRecord(
        name="test.security",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="authorization=%s",
        args=("Bearer ultra-secret-token",),
        exc_info=None,
    )
    record.authorization = "Bearer ultra-secret-token"
    record.context = "api_key=live_key_123"

    filt = SensitiveDataRedactionFilter()
    assert filt.filter(record) is True

    rendered = record.getMessage()
    assert "authorization=***redacted***" in rendered.lower()
    assert "ultra-secret-token" not in rendered
    assert record.authorization == "***REDACTED***"
    assert "api_key=***REDACTED***" in record.context
