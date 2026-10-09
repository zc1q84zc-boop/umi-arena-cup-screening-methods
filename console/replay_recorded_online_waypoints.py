"""Re-execute a recorded model waypoint stream for controller A/B diagnostics.

This consumes no model or recorded images. It is a deterministic command
replay for measuring simulator joint response, not a closed-loop evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


_path = Path(os.environ["UMI_REPLAY_AUDIT_PATH"])
_rows = [json.loads(line) for line in _path.read_text().splitlines()]
if not _rows or [row["step"] for row in _rows] != list(range(len(_rows))):
    raise ValueError("recorded online audit is not a sequential episode")
if any(len(row.get("substep_waypoints") or []) != 3 for row in _rows):
    raise ValueError("recorded online audit requires three 30 Hz waypoints per call")


def predict(observation, step, episode):
    if episode != 0 or not 0 <= step < len(_rows):
        raise ValueError("replay requested outside recorded model steps")
    return {"action_dt_s": 1 / 30,
            "waypoints": _rows[step]["substep_waypoints"],
            "execute_steps": 3}
