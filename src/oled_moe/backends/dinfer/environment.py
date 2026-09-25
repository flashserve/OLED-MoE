from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_dinfer_path(explicit_path: str | None = None) -> Path:
    value = explicit_path or os.environ.get("OLED_DINFER_PATH")
    if value is None:
        candidate = Path(__file__).resolve().parents[5] / "dInfer" / "python"
    else:
        candidate = Path(value)
    candidate = candidate.resolve()
    if not (candidate / "dinfer" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"dInfer Python source not found at {candidate}; set OLED_DINFER_PATH"
        )
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
    imported = sys.modules.get("dinfer")
    if imported is not None:
        imported_file = Path(getattr(imported, "__file__", "")).resolve()
        if candidate not in imported_file.parents:
            raise RuntimeError(
                f"dinfer was already imported from {imported_file}, expected {candidate}"
            )
    return candidate
