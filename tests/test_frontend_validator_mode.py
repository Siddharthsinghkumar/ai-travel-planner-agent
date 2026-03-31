from pathlib import Path
from types import SimpleNamespace

from validation.frontend_validator import FrontendValidator


class _DummyProc:
    def __init__(self, pid: int = 4321):
        self.pid = pid

    def poll(self):
        return None


def test_frontend_validator_dev_mode_starts_dev(monkeypatch, tmp_path):
    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return _DummyProc()

    monkeypatch.setattr("validation.frontend_validator.subprocess.Popen", fake_popen)

    validator = FrontendValidator(
        frontend_url="http://127.0.0.1:5173",
        frontend_dir=Path(tmp_path),
        frontend_server_mode="dev",
    )
    monkeypatch.setattr(validator, "_is_frontend_up", lambda: False)

    validator._start_frontend_server()

    assert popen_calls, "Expected frontend process start."
    cmd, kwargs = popen_calls[0]
    assert cmd[:3] == ["npm", "run", "dev"]
    assert "--host" in cmd and "--port" in cmd
    assert kwargs["cwd"] == str(Path(tmp_path))


def test_frontend_validator_preview_mode_builds_then_starts_preview(monkeypatch, tmp_path):
    run_calls = []
    popen_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return _DummyProc()

    monkeypatch.setattr("validation.frontend_validator.subprocess.run", fake_run)
    monkeypatch.setattr("validation.frontend_validator.subprocess.Popen", fake_popen)

    validator = FrontendValidator(
        frontend_url="http://127.0.0.1:5173",
        frontend_dir=Path(tmp_path),
        frontend_server_mode="preview",
    )
    monkeypatch.setattr(validator, "_is_frontend_up", lambda: False)

    validator._start_frontend_server()

    assert run_calls, "Expected preview build before server start."
    run_cmd, run_kwargs = run_calls[0]
    assert run_cmd == ["npm", "run", "build"]
    assert run_kwargs["cwd"] == str(Path(tmp_path))

    assert popen_calls, "Expected preview server process start."
    popen_cmd, popen_kwargs = popen_calls[0]
    assert popen_cmd[:3] == ["npm", "run", "preview"]
    assert "--host" in popen_cmd and "--port" in popen_cmd
    assert popen_kwargs["cwd"] == str(Path(tmp_path))
