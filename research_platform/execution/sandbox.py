import subprocess
import sys
import tempfile
import os
from typing import Dict, Any

class LiveSandbox:
    """
    Edison v4.0 Live Sandbox Execution (Subprocess-based for stability).
    Captures stdout, stderr and provides a clean interface for code verification.
    """

    def __init__(self):
        pass

    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Executes code in a subprocess and captures all output.
        """
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp.write(code.encode('utf-8'))
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\n--- Standard Error ---\n{result.stderr}"

            status = "ok" if result.returncode == 0 else "error"

            return {
                "status": status,
                "output": output.strip(),
                "code_executed": code
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "output": f"⚠️ Execution Timed Out after {timeout}s.",
                "code_executed": code
            }
        except Exception as e:
            return {
                "status": "error",
                "output": f"⚠️ Sandbox Error: {str(e)}",
                "code_executed": code
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def shutdown(self):
        pass

    def __del__(self):
        pass
