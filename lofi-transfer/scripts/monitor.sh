#!/bin/bash
# Overnight training monitor for the lofi diffusion pipeline.
#
# Every 5 min, checks on the remote training state and restarts if:
#   - tmux session `lofi_train` is gone
#   - train.log hasn't grown in >20 min (silent hang)
#   - GPU memory <5 GiB (training not loaded / crashed)
#   - No python3 training process alive on remote
#
# Restart strategy: find newest checkpoint*.pt on remote, restart from its step.
# If only checkpoint491668.pt exists (fresh run), restart from 491668.
#
# Pause hook: create /tmp/lofi_monitor_pause to disable restart logic temporarily
# (monitoring continues, but no auto-restart — useful if user is debugging manually).
#
# Launch via:
#   nohup caffeinate -ism bash scripts/monitor.sh </dev/null >/tmp/lofi_monitor.stdout 2>&1 & disown

SSH_HOST="root@50.217.254.161"
SSH_PORT=41822
SSH_OPTS="-p $SSH_PORT -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o StrictHostKeyChecking=accept-new"
REMOTE_BASE="/workspace/lofi-transfer"
LOG="/tmp/lofi_monitor.log"
STATUS="/tmp/lofi_monitor_status.txt"
PAUSE_FLAG="/tmp/lofi_monitor_pause"

STALL_THRESHOLD=1200    # 20 min of no log growth = stall
CHECK_INTERVAL=300      # check every 5 min
MIN_GPU_MEM=5000        # MiB; below this = training clearly not loaded
RESTART_COOLDOWN=600    # 10 min between successive restart attempts
LAST_RESTART=0

exec >>"$LOG" 2>&1

status() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" > "$STATUS"
}

status "monitor starting (pid=$$, interval=${CHECK_INTERVAL}s, stall=${STALL_THRESHOLD}s)"

restart_training() {
    local now=$(date +%s)
    local since_last=$((now - LAST_RESTART))
    if [ $since_last -lt $RESTART_COOLDOWN ]; then
        status "cooldown: last restart ${since_last}s ago (<${RESTART_COOLDOWN}s) — skipping"
        return 1
    fi

    status "RESTART: killing stale processes on remote"
    ssh $SSH_OPTS $SSH_HOST "pkill -9 -f 'train_diffusion.py' 2>/dev/null; tmux kill-session -t lofi_train 2>/dev/null; sleep 2; pkill -9 -f 'train_diffusion.py' 2>/dev/null" || true

    # Find newest checkpoint
    local latest_step
    latest_step=$(ssh $SSH_OPTS $SSH_HOST "ls $REMOTE_BASE/runs/lofi_v1/checkpoint*.pt 2>/dev/null | sed 's|.*checkpoint\([0-9]\+\)\.pt|\1|' | sort -n | tail -1")
    if [ -z "$latest_step" ]; then
        status "RESTART FAILED: no checkpoint found to restart from"
        return 1
    fi
    status "RESTART: resuming from checkpoint${latest_step}.pt"

    ssh $SSH_OPTS $SSH_HOST "cd $REMOTE_BASE/ctd_repo && tmux new-session -d -s lofi_train \
        'cd $REMOTE_BASE/ctd_repo && python3 train_diffusion.py \
            --name lofi_v1 \
            --db_path $REMOTE_BASE/data/lmdb/lofi \
            --dataset_type waveform \
            --gpu 0 \
            --bsize 64 \
            --restart $latest_step \
            --emb_model_path $REMOTE_BASE/pretrained/AE_real_instruments.pt \
            --out_path $REMOTE_BASE/runs/ 2>&1 | tee -a $REMOTE_BASE/runs/lofi_v1/train.log'"

    sleep 20
    local verify
    verify=$(ssh $SSH_OPTS $SSH_HOST "tmux list-sessions 2>&1 | grep lofi_train" || echo "")
    if [ -n "$verify" ]; then
        status "RESTART SUCCESS: tmux 'lofi_train' re-launched from step $latest_step"
        LAST_RESTART=$now
        return 0
    else
        status "RESTART FAILED: tmux session not found after launch attempt"
        return 1
    fi
}

while true; do
    # Probe remote state (tolerates ssh failures — treats as unknown, skips this cycle)
    remote_state=$(ssh $SSH_OPTS $SSH_HOST "
        tmux list-sessions 2>&1 | grep -q lofi_train && echo TMUX=alive || echo TMUX=dead
        echo LOG_MTIME=\$(stat -c%Y $REMOTE_BASE/runs/lofi_v1/train.log 2>/dev/null || echo 0)
        echo NOW=\$(date +%s)
        echo GPU_UTIL=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
        echo GPU_MEM=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print \$1}')
        echo PY=\$(pgrep -f 'train_diffusion.py' | head -1)
    " 2>&1) || remote_state=""

    if [ -z "$remote_state" ] || ! echo "$remote_state" | grep -q "^TMUX="; then
        status "ssh probe failed — remote unreachable? will retry"
        sleep $CHECK_INTERVAL
        continue
    fi

    tmux_state=$(echo "$remote_state" | grep "^TMUX=" | cut -d= -f2)
    log_mtime=$(echo "$remote_state" | grep "^LOG_MTIME=" | cut -d= -f2)
    now_remote=$(echo "$remote_state" | grep "^NOW=" | cut -d= -f2)
    gpu_util=$(echo "$remote_state" | grep "^GPU_UTIL=" | cut -d= -f2)
    gpu_mem=$(echo "$remote_state" | grep "^GPU_MEM=" | cut -d= -f2)
    py_pid=$(echo "$remote_state" | grep "^PY=" | cut -d= -f2)
    age=$((now_remote - log_mtime))

    status "check: tmux=$tmux_state py_pid=${py_pid:-none} log_age=${age}s gpu=${gpu_util}% mem=${gpu_mem}MiB"

    # Respect pause flag
    if [ -f "$PAUSE_FLAG" ]; then
        status "paused (by $PAUSE_FLAG) — skipping restart check"
        sleep $CHECK_INTERVAL
        continue
    fi

    # Failure detection, most urgent first
    if [ "$tmux_state" = "dead" ] && [ -z "$py_pid" ]; then
        status "FAIL: no tmux + no python training process"
        restart_training
    elif [ -z "$py_pid" ]; then
        status "FAIL: python training process missing (tmux may be shell-only now)"
        restart_training
    elif [ "$age" -gt "$STALL_THRESHOLD" ]; then
        status "FAIL: train.log not updated in ${age}s (>${STALL_THRESHOLD}s) — stuck"
        restart_training
    elif [ "$gpu_mem" -lt "$MIN_GPU_MEM" ]; then
        status "FAIL: GPU memory ${gpu_mem}MiB < ${MIN_GPU_MEM}MiB — training not loaded"
        restart_training
    else
        : # healthy
    fi

    sleep $CHECK_INTERVAL
done
