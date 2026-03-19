import pytest

from core.env_config import (
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_str,
    is_env_set,
    parse_csv_env,
)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_get_env_bool_truthy(monkeypatch, value):
    monkeypatch.setenv("BOOL_FLAG", value)
    assert get_env_bool("BOOL_FLAG", default=False) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_get_env_bool_falsy(monkeypatch, value):
    monkeypatch.setenv("BOOL_FLAG", value)
    assert get_env_bool("BOOL_FLAG", default=True) is False


def test_get_env_bool_invalid_uses_default_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("BOOL_FLAG", "sometimes")
    caplog.set_level("WARNING")
    assert get_env_bool("BOOL_FLAG", default=True) is True
    assert "Invalid boolean env BOOL_FLAG" in caplog.text


def test_get_env_int_and_float_invalid_fallback(monkeypatch, caplog):
    monkeypatch.setenv("INT_FLAG", "abc")
    monkeypatch.setenv("FLOAT_FLAG", "xyz")
    caplog.set_level("WARNING")

    assert get_env_int("INT_FLAG", 42) == 42
    assert get_env_float("FLOAT_FLAG", 3.5) == 3.5
    assert "Invalid integer env INT_FLAG" in caplog.text
    assert "Invalid float env FLOAT_FLAG" in caplog.text


def test_parse_csv_env_and_get_env_str(monkeypatch):
    monkeypatch.setenv("CSV_FLAG", " alpha, beta ,,gamma ")
    monkeypatch.setenv("STR_FLAG", "  value  ")

    assert parse_csv_env("CSV_FLAG") == ["alpha", "beta", "gamma"]
    assert parse_csv_env("CSV_MISSING", default=["x", " y "]) == ["x", "y"]
    assert get_env_str("STR_FLAG") == "value"
    assert get_env_str("STR_MISSING", default="fallback") == "fallback"


def test_is_env_set(monkeypatch):
    monkeypatch.delenv("UNSET_FLAG", raising=False)
    assert is_env_set("UNSET_FLAG") is False

    monkeypatch.setenv("UNSET_FLAG", "   ")
    assert is_env_set("UNSET_FLAG") is False

    monkeypatch.setenv("UNSET_FLAG", "1")
    assert is_env_set("UNSET_FLAG") is True
