#!/usr/bin/env bash
#
# Local equivalent of the GitHub CI checks (rust.yml + webui-wasm.yml).
# Run from the repo root:  ./scripts/check-all.sh
#
# Flags:
#   --no-test    Skip test steps (faster iteration on lint/build issues)
#   --no-wasm    Skip the WASM/webui checks (requires wasm32 target + wasm-pack)
#
set -euo pipefail

RUN_TESTS=1
RUN_WASM=1
for arg in "$@"; do
  case "$arg" in
    --no-test)  RUN_TESTS=0 ;;
    --no-wasm)  RUN_WASM=0 ;;
    *)          echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

BOLD='\033[1m'
GREEN='\033[1;32m'
RED='\033[1;31m'
RESET='\033[0m'

pass=0
fail=0

step() {
  echo -e "\n${BOLD}── $1${RESET}"
}

run() {
  if "$@"; then
    echo -e "${GREEN}  ✓ passed${RESET}"
    pass=$((pass + 1))
  else
    echo -e "${RED}  ✗ FAILED${RESET}"
    fail=$((fail + 1))
  fi
}

# ── Compute package filters (exclude /libs, matching CI) ──────────────
META=$(cargo metadata --no-deps --format-version 1)
FMT_PKGS=$(echo "$META" | jq -r '.packages[] | select(.manifest_path | test("/libs/") | not) | .name' | while read -r p; do printf " -p %s" "$p"; done)

# ── rust.yml checks ──────────────────────────────────────────────────

step "Check formatting (exclude /libs)"
run cargo fmt $FMT_PKGS -- --check

step "Clippy whisper-tensor (all features, deny warnings)"
run cargo clippy -p whisper-tensor --no-deps --all-features -- -D warnings

step "Clippy whisper-tensor-server (all features, deny warnings)"
run cargo clippy -p whisper-tensor-server --no-deps --all-features -- -D warnings

step "Clippy whisper-tensor-import (all features, deny warnings)"
run cargo clippy -p whisper-tensor-import --no-deps --all-features -- -D warnings

step "Build whisper-tensor (no features)"
run cargo build -p whisper-tensor --lib --locked

step "Build whisper-tensor (all features)"
run cargo build -p whisper-tensor --lib --locked --all-features

step "Build whisper-tensor-server (no features)"
run cargo build -p whisper-tensor-server --locked

step "Build whisper-tensor-server (all features)"
run cargo build -p whisper-tensor-server --locked --all-features

if [[ "$RUN_TESTS" -eq 1 ]]; then
  step "Test whisper-tensor (all features)"
  run cargo test -p whisper-tensor --all-features --locked

  step "Test whisper-tensor-server (all features)"
  run cargo test -p whisper-tensor-server --all-features --locked
fi

# ── webui-wasm.yml checks ────────────────────────────────────────────

if [[ "$RUN_WASM" -eq 1 ]]; then
  step "Check formatting (webui)"
  run cargo fmt -p whisper-tensor-ui -- --check

  step "Clippy webui (wasm32, deny warnings)"
  run env RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
    cargo clippy -p whisper-tensor-ui --no-deps --target wasm32-unknown-unknown -- -D warnings

  if command -v wasm-pack &>/dev/null; then
    step "Build webui (wasm-pack)"
    run wasm-pack build crates/whisper-tensor-ui --target web --locked
  else
    echo -e "\n${BOLD}── Skipping wasm-pack build (wasm-pack not installed)${RESET}"
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────

echo -e "\n${BOLD}════════════════════════════════════════${RESET}"
if [[ "$fail" -eq 0 ]]; then
  echo -e "${GREEN}All $pass checks passed.${RESET}"
else
  echo -e "${RED}$fail of $((pass + fail)) checks FAILED.${RESET}"
  exit 1
fi
