#!/usr/bin/env python3
"""Build and compile a draft autoregressive core model for ANE benchmarks.

Model input:  (1, 1, 768) float32 embedding
Model output: (1, 1, 32000) float32 logits
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
if sys.path and sys.path[0] == SCRIPT_DIR:
    sys.path.pop(0)

SITE = pathlib.Path(__file__).resolve().parents[1] / ".venv-coreml" / "lib" / "python3.9" / "site-packages"
if SITE.exists() and str(SITE) not in sys.path:
    sys.path.insert(0, str(SITE))

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class DraftBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, n_heads: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

        self.norm1 = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        self.norm2 = RMSNorm(hidden)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        h = self.norm1(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(b, t, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(b, t, self.n_heads, self.head_dim)

        # Single-token autoregressive decode has trivial softmax; keep q/k/v projection cost.
        a = ((q + k + v) * (1.0 / 3.0)).contiguous().view(b, t, self.hidden)
        x = x + self.o_proj(a)

        h2 = self.norm2(x)
        ffn = torch.nn.functional.silu(self.gate_proj(h2)) * self.up_proj(h2)
        x = x + self.down_proj(ffn)
        return x


class DraftCoreModel(nn.Module):
    def __init__(self, hidden: int, intermediate: int, n_layers: int, n_heads: int, vocab: int, emb_table: torch.Tensor) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DraftBlock(hidden=hidden, intermediate=intermediate, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(emb_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


def build(out_dir: pathlib.Path,
          hidden: int = 768,
          intermediate: int = 3072,
          n_layers: int = 6,
          n_heads: int = 12,
          vocab: int = 32000) -> None:
    torch.manual_seed(0)

    emb_table = torch.randn(vocab, hidden, dtype=torch.float32) * 0.02
    model = DraftCoreModel(
        hidden=hidden,
        intermediate=intermediate,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab=vocab,
        emb_table=emb_table,
    ).eval()

    sample = torch.randn(1, 1, hidden, dtype=torch.float32)
    traced = torch.jit.trace(model, sample)

    mlmodel = ct.convert(
        traced,
        convert_to="neuralnetwork",
        minimum_deployment_target=ct.target.macOS11,
        inputs=[ct.TensorType(name="x", shape=(1, 1, hidden), dtype=np.float32)],
    )

    pkg = out_dir / "draft125m_full_core.mlpackage"
    modelc = out_dir / "draft125m_full_core.mlmodelc"
    emb_bin = out_dir / "draft125m_full_embed_f32.bin"
    meta = out_dir / "draft125m_full_embed_meta.txt"

    if pkg.exists():
        subprocess.run(["rm", "-rf", str(pkg)], check=True)
    if modelc.exists():
        subprocess.run(["rm", "-rf", str(modelc)], check=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(pkg))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)], check=True)

    emb = emb_table.detach().cpu().numpy().astype(np.float32, copy=False)
    emb.tofile(str(emb_bin))
    meta.write_text(
        f"vocab={vocab}\nhidden={hidden}\nrows={vocab}\ncols={hidden}\ndtype=float32\nfile={emb_bin.name}\n",
        encoding="utf-8",
    )

    print(f"ok: {pkg}")
    print(f"ok: {modelc}")
    print(f"ok: {emb_bin}")
    print(f"ok: {meta}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn", help="output directory")
    args = ap.parse_args()

    build(pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
