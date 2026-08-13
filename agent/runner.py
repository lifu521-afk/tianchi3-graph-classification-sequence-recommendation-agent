from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _command(spec: dict[str, Any], root: Path, run_dir: Path) -> list[str]:
    script = root / str(spec["script"])
    args = [str(value).replace("{run_dir}", str(run_dir)) for value in spec.get("args", [])]
    return [sys.executable, str(script), *args]


def run_experiment(
    spec: dict[str, Any],
    root: Path,
    run_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = _command(spec, root, run_dir)
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        status = "completed" if completed.returncode == 0 else "failed"
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        return_code = None
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    duration = time.monotonic() - started
    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (run_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": status,
        "return_code": return_code,
        "duration_seconds": round(duration, 3),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
