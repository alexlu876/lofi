#!/bin/bash
# Orchestrates end-to-end launch after LMDB build completes:
#   1. Wait for split_to_lmdb.py to exit
#   2. Verify LMDB structure
#   3. Measure z.std (informational)
#   4. Rsync LMDB + repo + pretrained + run dir to vast.ai
#   5. Install deps on remote
#   6. Smoke test (2-min run, check for loss)
#   7. Launch full training in tmux
#
# Status tracked in /tmp/lofi_status.txt (one-line) and /tmp/lofi_auto_launch.log (full log).

SSH_HOST="root@50.217.254.161"
SSH_PORT=41822
SSH_OPTS="-p $SSH_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15 -o TCPKeepAlive=yes -o StrictHostKeyChecking=accept-new"
LOCAL_BASE="/Users/alex/lu/git/lofi/lofi-transfer"
REMOTE_BASE="/workspace/lofi-transfer"
PYTHON="/Users/alex/lu/git/lofi/lofi/bin/python3"
LOG="/tmp/lofi_auto_launch.log"
STATUS="/tmp/lofi_status.txt"

exec >>"$LOG" 2>&1

# --- Sleep resistance: start caffeinate so the Mac stays awake as long as we run ---
# -i prevents idle sleep, -s prevents system sleep on AC power, -m prevents disk sleep
caffeinate -ism &
CAFF_PID=$!
trap "kill $CAFF_PID 2>/dev/null; echo '[cleanup] caffeinate released'" EXIT INT TERM
echo "$(date '+%Y-%m-%d %H:%M:%S') caffeinate running as PID $CAFF_PID (system will not sleep)"

status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee "$STATUS"
}

# --- rsync with retry loop (handles transient network drops / sleep-induced breakage) ---
# Exponential-ish backoff, capped at 5 min. 30 attempts gives up to ~90 min of persistent-failure tolerance.
rsync_retry() {
    local max=30
    local delay
    for i in $(seq 1 $max); do
        rsync "$@" && return 0
        # Backoff: 30s, 60s, 90s, 120s, ... capped at 300s
        delay=$((i * 30))
        [ $delay -gt 300 ] && delay=300
        status "rsync attempt $i/$max failed (exit $?); retrying in ${delay}s..."
        sleep $delay
    done
    return 1
}
step() {
    echo ""
    echo "=========================================================="
    status "$1"
    echo "=========================================================="
}
fatal() {
    status "FATAL: $1"
    exit 1
}

step "auto_launch starting. Watching LMDB build..."

# ---- Phase 1: Wait for LMDB build to exit (polls every 3 min) ----
while pgrep -f "split_to_lmdb.py" >/dev/null; do
    status "LMDB build still running; sleeping 180s..."
    sleep 180
done
status "LMDB build process exited"

# ---- Phase 2: Verify LMDB ----
step "Verifying LMDB integrity"
if ! [ -d "$LOCAL_BASE/data/lmdb/lofi" ]; then
    fatal "LMDB directory does not exist at $LOCAL_BASE/data/lmdb/lofi"
fi
"$PYTHON" -c "
import lmdb, sys
env = lmdb.open('$LOCAL_BASE/data/lmdb/lofi', readonly=True, lock=False)
with env.begin() as txn:
    n = txn.stat()['entries']
    print(f'LMDB entries: {n}')
    assert n > 1000, f'too few entries: {n}'
print('LMDB sanity OK')
" || fatal "LMDB verification failed"

LMDB_SIZE=$(du -sh "$LOCAL_BASE/data/lmdb/lofi" | awk '{print $1}')
status "LMDB verified — size $LMDB_SIZE"

# ---- Phase 3: Measure z.std (informational, non-blocking) ----
step "Measuring lofi z.std via AE (best-effort)"
"$PYTHON" "$LOCAL_BASE/scripts/measure_z_std.py" || status "z.std measurement FAILED (continuing anyway)"

# ---- Phase 4: Rsync to remote ----
step "Rsync pretrained bundle (small, ~2 GB)"
ssh $SSH_OPTS $SSH_HOST "mkdir -p $REMOTE_BASE/pretrained $REMOTE_BASE/ctd_repo $REMOTE_BASE/runs $REMOTE_BASE/data/lmdb"
rsync_retry -az --partial -e "ssh $SSH_OPTS" \
    "$LOCAL_BASE/pretrained/" "$SSH_HOST:$REMOTE_BASE/pretrained/" \
    || fatal "pretrained rsync failed"

step "Rsync ctd_repo + patches"
rsync_retry -az --partial --delete --exclude='.git' --exclude='__pycache__' --exclude='notebooks' -e "ssh $SSH_OPTS" \
    "$LOCAL_BASE/ctd_repo/" "$SSH_HOST:$REMOTE_BASE/ctd_repo/" \
    || fatal "ctd_repo rsync failed"

step "Rsync run dir (checkpoint491668.pt + config.gin)"
rsync_retry -az --partial -e "ssh $SSH_OPTS" \
    "$LOCAL_BASE/runs/lofi_v1/" "$SSH_HOST:$REMOTE_BASE/runs/lofi_v1/" \
    || fatal "runs rsync failed"

step "Rsync LMDB (~56 GB dense — LMDB compacted so no sparse handling needed)"
# Source data.mdb is now dense (compacted from sparse map_size=120GB into actual 56GB).
# Flags:
#   --inplace     : write directly to data.mdb (no .tmp file, no orphans on retry failure)
#   --timeout=300 : rsync fails after 5 min no-data (catches silent stalls)
#   -az           : archive + gzip compression
rsync_retry -az --inplace --timeout=300 -e "ssh $SSH_OPTS" \
    "$LOCAL_BASE/data/lmdb/lofi/" "$SSH_HOST:$REMOTE_BASE/data/lmdb/lofi/" \
    || fatal "LMDB rsync failed"
status "Rsync complete"

# ---- Phase 5: Install deps on remote ----
step "Install deps on remote"
ssh $SSH_OPTS $SSH_HOST "cd $REMOTE_BASE/ctd_repo && pip install -q -r requirements.txt 2>&1 | tail -3" \
    || status "Pip install had warnings (check log)"

# Sanity check critical imports
ssh $SSH_OPTS $SSH_HOST "cd $REMOTE_BASE && python3 -c '
import torch, cqt_pytorch, cached_conv, gin, lmdb, einops_exts
print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())
'" || fatal "Critical import check failed on remote"

# ---- Phase 6: Smoke test (launch for 90s, kill, check for loss) ----
step "Smoke test — 90-second training run"
ssh $SSH_OPTS $SSH_HOST "cd $REMOTE_BASE/ctd_repo && timeout 90 python3 train_diffusion.py \
    --name lofi_v1 \
    --db_path $REMOTE_BASE/data/lmdb/lofi \
    --dataset_type waveform \
    --gpu 0 \
    --bsize 64 \
    --restart 491668 \
    --emb_model_path $REMOTE_BASE/pretrained/AE_real_instruments.pt \
    --out_path $REMOTE_BASE/runs/ > $REMOTE_BASE/runs/lofi_v1/smoke.log 2>&1; echo EXIT=\$?" \
    | tee -a "$LOG"

# Verify smoke log has a loss line (not NaN, not empty)
status "Checking smoke test output for valid loss..."
SMOKE_CHECK=$(ssh $SSH_OPTS $SSH_HOST "
if ! [ -f $REMOTE_BASE/runs/lofi_v1/smoke.log ]; then echo MISSING; exit 1; fi
if grep -qE 'Traceback|RuntimeError|ValueError|KeyError|AttributeError|TypeError|nan,|NaN' $REMOTE_BASE/runs/lofi_v1/smoke.log; then echo HAD_ERROR; exit 1; fi
# Training outputs 'loss=0.XXX' in tqdm progress bar. Also accept 'Diffusion loss' (tensorboard scalar) or 'Loss/' (scalar prefix).
if grep -qE 'loss=[0-9]|Diffusion loss|Loss/' $REMOTE_BASE/runs/lofi_v1/smoke.log; then echo OK; else echo NO_LOSS; fi
")
status "Smoke result: $SMOKE_CHECK"
if [ "$SMOKE_CHECK" != "OK" ]; then
    ssh $SSH_OPTS $SSH_HOST "tail -30 $REMOTE_BASE/runs/lofi_v1/smoke.log" | tee -a "$LOG"
    fatal "Smoke test did not produce clean loss — NOT launching training"
fi

# ---- Phase 7: Launch full training in tmux ----
step "Launching real training in tmux session 'lofi_train'"
ssh $SSH_OPTS $SSH_HOST "command -v tmux >/dev/null || apt-get install -y tmux 2>&1 | tail -2"

ssh $SSH_OPTS $SSH_HOST "cd $REMOTE_BASE/ctd_repo && tmux kill-session -t lofi_train 2>/dev/null; \
    tmux new-session -d -s lofi_train \
    'cd $REMOTE_BASE/ctd_repo && python3 train_diffusion.py \
        --name lofi_v1 \
        --db_path $REMOTE_BASE/data/lmdb/lofi \
        --dataset_type waveform \
        --gpu 0 \
        --bsize 64 \
        --restart 491668 \
        --emb_model_path $REMOTE_BASE/pretrained/AE_real_instruments.pt \
        --out_path $REMOTE_BASE/runs/ 2>&1 | tee -a $REMOTE_BASE/runs/lofi_v1/train.log'"

sleep 10
PROC=$(ssh $SSH_OPTS $SSH_HOST "tmux list-sessions 2>&1 | grep lofi_train" || echo "")
if [ -z "$PROC" ]; then
    fatal "Training tmux session NOT found after launch"
fi

status "SUCCESS — training launched in tmux session 'lofi_train'"

cat >>"$LOG" <<EOF

============================================================
SUCCESS — auto_launch complete at $(date '+%Y-%m-%d %H:%M:%S')
============================================================
Training is running in tmux session 'lofi_train' on the vast.ai box.

Attach:
  ssh -p $SSH_PORT $SSH_HOST 'tmux a -t lofi_train'

Monitor logs:
  ssh -p $SSH_PORT $SSH_HOST 'tail -f $REMOTE_BASE/runs/lofi_v1/train.log'

GPU/memory:
  ssh -p $SSH_PORT $SSH_HOST 'nvidia-smi'

Checkpoints save every 50k steps at:
  $REMOTE_BASE/runs/lofi_v1/checkpoint*.pt

First audio sample at step ~500,668 (10k past restart = 501,668... approximately).
Actually: steps_valid=10000 counts from fine-tune step 1, so first val around restart+10k = ~501,668.

LMDB size: $LMDB_SIZE

z.std measurement: /tmp/lofi_z_std.json
============================================================
EOF

cat "$LOG" | tail -40
