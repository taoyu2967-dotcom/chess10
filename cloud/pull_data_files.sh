#!/usr/bin/env bash
# 逐文件拉取 training/data + data_az（单流断线只重当前文件）
SSH="ssh -i $HOME/.ssh/autodl_chess10 -p 47010 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
SCP="scp -i $HOME/.ssh/autodl_chess10 -P 47010 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q"
R=root@region-42.seetacloud.com
RB=/root/autodl-tmp/chess
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # 路径中枢约定：脚本位置推导
DEST="$ROOT/cloud_pull/cloud_training_data"
LOG="$ROOT/cloud_pull/datapull.log"
mkdir -p "$DEST/data" "$DEST/data_az"
{
  echo "=== data 逐文件拉取 $(date '+%F %T') ==="
  FILES=$($SSH $R "cd $RB && ls training/data/*.f32 training/data_az/*.f32")
  echo "文件数: $(echo "$FILES" | wc -l)"
  fail=0
  for f in $FILES; do
    rel=${f#training/}
    case "$rel" in data/*) sub=data;; data_az/*) sub=data_az;; esac
    base=$(basename "$f")
    tgt="$DEST/$sub/$base"
    rsize=$($SSH $R "stat -c %s $RB/$f" 2>/dev/null)
    lsize=$(stat -c %s "$tgt" 2>/dev/null || echo 0)
    if [ "$rsize" = "$lsize" ] && [ -n "$rsize" ]; then echo "SKIP $rel ($rsize)"; continue; fi
    if $SCP "$R:$RB/$f" "$tgt" 2>>"$LOG"; then
      echo "OK   $rel ($(stat -c %s "$tgt"))"
    else
      echo "FAIL $rel"; fail=$((fail+1))
    fi
  done
  echo "=== 完成 $(date '+%F %T') 失败 $fail ==="
} > "$LOG" 2>&1
