#!/usr/bin/env bash
set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
DIR_TO_SYNC='CelebV-HQ'
USER='web'
HOST='192.168.0.48'

# ── choose mode & destination ────────────────────────────────────────────────
MODE="${1:-dev}"          # default when no arg is given

if [[ "$MODE" == "production" ]]; then
    DEST_BASE='/home/web/partha'          # ← production path (no /dev)
else
    MODE='dev'                            # normal watch-loop
    DEST_BASE='/home/web/partha/dev'
fi

DEST_PATH="$DEST_BASE/$DIR_TO_SYNC"
EXCLUDES=( --exclude='.*' --exclude='uploads' --exclude='*.pyc' --exclude='__pycache__/')

# ── helper: do one rsync ─────────────────────────────────────────────────────
sync_once () {
    rsync -av "${EXCLUDES[@]}" "../${DIR_TO_SYNC}/" \
          "${USER}@${HOST}:${DEST_PATH}"
}

# ── run ──────────────────────────────────────────────────────────────────────
echo "Mode: $MODE → syncing to ${USER}@${HOST}:${DEST_PATH}"

if [[ "$MODE" == "production" ]]; then
    sync_once
    echo "✔ pushed to production"
    exit 0
fi

# dev-mode: watch & repeat
while inotifywait -r --exclude '/\.' "../${DIR_TO_SYNC}"; do
    sync_once
done
