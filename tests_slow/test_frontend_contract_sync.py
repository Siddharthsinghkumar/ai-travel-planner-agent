import subprocess
import sys
from pathlib import Path


def test_frontend_contract_artifacts_are_in_sync() -> None:
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "scripts" / "sync_frontend_contract.py"), "--check"]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout + "\n" + result.stderr).strip()
