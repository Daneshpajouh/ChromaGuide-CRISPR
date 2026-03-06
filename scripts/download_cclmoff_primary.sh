#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/data/public_benchmarks/off_target/primary_cclmoff"
DEST_FILE="$DEST_DIR/09212024_CCLMoff_dataset.csv"
PARTIAL_FILE="$DEST_FILE.partial"
URL="https://ndownloader.figshare.com/files/49344577"
EXPECTED_BYTES=714692329
EXPECTED_MD5="2a9be5c69a89c8eee3fdef0c03efae3a"

file_size_bytes() {
  local path="$1"
  if stat -c%s "$path" >/dev/null 2>&1; then
    stat -c%s "$path"
  else
    stat -f%z "$path"
  fi
}

file_md5() {
  local path="$1"
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$path" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$path"
  else
    python3 - "$path" <<'PY'
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1])
h = hashlib.md5()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
  fi
}

mkdir -p "$DEST_DIR"

if [ -f "$DEST_FILE" ]; then
  ACTUAL_BYTES=$(file_size_bytes "$DEST_FILE")
  if [ "$ACTUAL_BYTES" -eq "$EXPECTED_BYTES" ]; then
    echo "Complete file already present: $DEST_FILE"
    exit 0
  fi
  mv "$DEST_FILE" "$PARTIAL_FILE"
fi

if [ -f "$PARTIAL_FILE" ]; then
  curl -fL -C - "$URL" -o "$PARTIAL_FILE"
else
  curl -fL "$URL" -o "$PARTIAL_FILE"
fi

ACTUAL_BYTES=$(file_size_bytes "$PARTIAL_FILE")
if [ "$ACTUAL_BYTES" -eq "$EXPECTED_BYTES" ]; then
  mv "$PARTIAL_FILE" "$DEST_FILE"
  ACTUAL_MD5=$(file_md5 "$DEST_FILE")
  echo "Downloaded complete file: $DEST_FILE"
  echo "MD5: $ACTUAL_MD5"
  if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "MD5 mismatch: expected $EXPECTED_MD5" >&2
    exit 1
  fi
else
  echo "Partial download retained: $PARTIAL_FILE ($ACTUAL_BYTES / $EXPECTED_BYTES bytes)"
fi
