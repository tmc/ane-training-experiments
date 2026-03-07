#!/usr/bin/env python3
"""Build and compile FFN CoreML models for ANE benchmarking."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

# Avoid shadowing stdlib modules with training/*.py (for example training/tokenize.py).
SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
if sys.path and sys.path[0] == SCRIPT_DIR:
    sys.path.pop(0)

# Use coremltools from project venv, torch from system site-packages.
SITE = pathlib.Path(__file__).resolve().parents[1] / ".venv-coreml" / "lib" / "python3.9" / "site-packages"
if SITE.exists() and str(SITE) not in sys.path:
    sys.path.insert(0, str(SITE))

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def build_mlpackage(out_dir: pathlib.Path, name: str, hidden: int, intermediate: int, seq_max: int = 2048) -> pathlib.Path:
    torch.manual_seed(0)
    model = FFN(hidden=hidden, intermediate=intermediate).eval()
    sample = torch.randn(1, 1, hidden, dtype=torch.float32)
    traced = torch.jit.trace(model, sample)

    mlmodel = ct.convert(
        traced,
        convert_to="neuralnetwork",
        minimum_deployment_target=ct.target.macOS11,
        inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, seq_max), hidden), dtype=np.float32)],
    )

    pkg = out_dir / f"{name}.mlpackage"
    if pkg.exists():
        if pkg.is_dir():
            subprocess.run(["rm", "-rf", str(pkg)], check=True)
        else:
            pkg.unlink()
    mlmodel.save(str(pkg))
    return pkg


def compile_mlpackage(pkg: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)], check=True)
    return out_dir / f"{pkg.stem}.mlmodelc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn", help="output directory")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("llama3b_ffn", 3072, 8192),
        ("draft125m_ffn", 768, 3072),
    ]

    for name, hidden, intermediate in specs:
        print(f"building {name} hidden={hidden} intermediate={intermediate}")
        pkg = build_mlpackage(out_dir, name, hidden, intermediate)
        modelc = compile_mlpackage(pkg, out_dir)
        print(f"ok: {pkg}")
        print(f"ok: {modelc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
