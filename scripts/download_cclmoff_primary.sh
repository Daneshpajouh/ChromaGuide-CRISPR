#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/data/public_benchmarks/off_target/primary_cclmoff"
DEST_FILE="$DEST_DIR/09212024_CCLMoff_dataset.csv"
PARTIAL_FILE="$DEST_FILE.partial"
URL="https://ndownloader.figshare.com/files/49344577"
EXPECTED_BYTES=714692329
EXPECTED_MD5="2a9be5c69a89c8eee3fdef0c03efae3a"

mkdir -p "$DEST_DIR"

if [ -f "$DEST_FILE" ]; then
  ACTUAL_BYTES=$(stat -f%z "$DEST_FILE")
  if [ "$ACTUAL_BYTES" -eq "$EXPECTED_BYTES" ]; then
    echo "Complete file already present: $DEST_FILE"
    exit 0
  fi
  mv "$DEST_FILE" "$PARTIAL_FILE"
fi

if [ -f "$PARTIAL_FILE" ]; then
  curl -L -C - "$URL" -o "$PARTIAL_FILE"
else
  curl -L "$URL" -o "$PARTIAL_FILE"
fi

ACTUAL_BYTES=$(stat -f%z "$PARTIAL_FILE")
if [ "$ACTUAL_BYTES" -eq "$EXPECTED_BYTES" ]; then
  mv "$PARTIAL_FILE" "$DEST_FILE"
  ACTUAL_MD5=$(md5 -q "$DEST_FILE")
  echo "Downloaded complete file: $DEST_FILE"
  echo "MD5: $ACTUAL_MD5"
  if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "MD5 mismatch: expected $EXPECTED_MD5" >&2
    exit 1
  fi
else
  echo "Partial download retained: $PARTIAL_FILE ($ACTUAL_BYTES / $EXPECTED_BYTES bytes)"
fi
