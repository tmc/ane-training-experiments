#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/training/test_chaining_suite.m"
BIN="$ROOT/training/test_chaining_suite"

clang -Wall -Wextra -O2 -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -F/System/Library/PrivateFrameworks \
  -framework AppleNeuralEngine \
  -ldl \
  "$SRC" -o "$BIN"

exec "$BIN" "$@"
