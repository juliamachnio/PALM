import subprocess
import sys
from pathlib import Path

def test_direct_schedule_cli_rejects_invalid_threshold_order(tmp_path):
    script = Path(__file__).resolve().parents[1] / "export_schedule.py"
    completed = subprocess.run([sys.executable, str(script), "--output", str(tmp_path / "x.json"), "--switch-1", "300", "--switch-2", "100"], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "strictly increasing" in completed.stderr
