# Release Checklist

This checklist is for cutting a separate release with parity and coverage evidence.

## 1) Clean working tree for release branch

```bash
git status --short
```

Only keep intended tracked changes for release.

## 2) Run readiness gate

```bash
./scripts/release_readiness.sh
```

This script enforces:

- Go package coverage smoke for:
  - `./ane/bridge`
  - `./ane/clientmodel`
  - `./ane/storiestrainer`
  - `./cmd/train-stories-ane-go`
- Strict C/ObjC vs Go parity suite (high-util profile).
- Seq map threshold sweep (`0x1D` boundary check).

Default strict thresholds:

- `go_vs_c_train_step_ratio <= 1.08`
- `go_ane_util_pct >= 8.0`
- `c_ane_util_pct >= 8.0`

## 3) Inspect readiness artifacts

The script prints a run directory:

- `release_readiness_dir=/tmp/ane_release_readiness/run_<timestamp>`

Core files:

- `report.txt`
- `go_coverage_summary.txt`
- `parity_suite.log`

Parity suite artifacts are under:

- `/tmp/ane_parity_suite/suite_<timestamp>/`

## 4) Optional tighter gates for release candidates

```bash
ANE_RELEASE_MAX_GO_VS_C_RATIO=1.05 \
ANE_RELEASE_MIN_GO_ANE_UTIL=10.0 \
ANE_RELEASE_MIN_C_ANE_UTIL=10.0 \
./scripts/release_readiness.sh
```

## 5) Publish summary with release

Include these fields from parity summary:

- `avg_c_train_step_ms`
- `avg_go_train_step_ms`
- `go_vs_c_train_step_ratio`
- `c_ane_util_pct`
- `go_ane_util_pct`

And include seq floor result:

- expected `seq floor = 16` on current host/compiler path.
