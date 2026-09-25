from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

PACKAGES = (
    "torch",
    "transformers",
    "safetensors",
    "vllm",
    "fastapi",
    "uvicorn",
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the OLED-MoE runtime environment")
    parser.add_argument(
        "--dinfer-path",
        type=Path,
        help="path containing the dinfer Python package; defaults to OLED_DINFER_PATH",
    )
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    errors: list[str] = []
    print(f"python={platform.python_version()} executable={sys.executable}")

    for package in PACKAGES:
        print(f"{package}={_version(package)}")

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        errors.append("PyTorch is not installed")
    else:
        print(f"torch_cuda_runtime={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            errors.append("CUDA was requested but torch.cuda.is_available() is false")

    raw_dinfer_path = args.dinfer_path or os.environ.get("OLED_DINFER_PATH")
    if raw_dinfer_path is None:
        errors.append("set OLED_DINFER_PATH or pass --dinfer-path")
    else:
        dinfer_path = Path(raw_dinfer_path).expanduser().resolve()
        init_file = dinfer_path / "dinfer" / "__init__.py"
        print(f"dinfer_path={dinfer_path}")
        if not init_file.is_file():
            errors.append(f"dInfer Python package was not found at {init_file}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("environment_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
