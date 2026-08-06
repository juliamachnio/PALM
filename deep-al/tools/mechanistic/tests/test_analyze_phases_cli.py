import subprocess
import sys
from pathlib import Path


def test_global_phase_analysis_cli_help_works():
    script = Path(__file__).resolve().parents[1] / "analyze_phases.py"
    completed = subprocess.run([sys.executable, str(script), "--help"], check=True, capture_output=True, text=True)
    assert "global DP/BIC" in completed.stdout
