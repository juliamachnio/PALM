import json
import subprocess
import sys
from pathlib import Path

def test_direct_schedule_cli_writes_explicit_thresholds(tmp_path):
    script, output = Path(__file__).resolve().parents[1] / "export_schedule.py", tmp_path / "schedule.json"
    subprocess.run([sys.executable, str(script), "--output", str(output), "--switch-1", "100", "--switch-2", "300"], check=True)
    payload = json.loads(output.read_text())
    assert payload["thresholds"] == [100.0, 300.0]
    assert payload["selection"] == "predefined_thresholds"
