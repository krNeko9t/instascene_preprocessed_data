from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_existing_object_results(output_json: Path) -> Dict[int, Dict[str, Any]]:
    if not output_json.exists():
        return {}
    with output_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    existing: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                continue
            existing[int(item["id"])] = item
    return existing


def append_error_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_results_list(output_json: Path, rows: list[dict[str, Any]]) -> None:
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
