#!/usr/bin/env bash
# 后台等待到「明天 03:00」再执行取回；日志写在 cloud_pull/waiter.log，便于事后核对
DEST="D:/data/新建文件夹/chess_game/cloud_pull"
PULL="D:/data/新建文件夹/chess_game/cloud/pull_results.sh"
LOG="$DEST/waiter.log"
mkdir -p "$DEST"

echo "[waiter] 启动 $(date '+%F %T')  PID=$$" >> "$LOG"
TGT=$(date -d "tomorrow 03:00" +%s)
echo "[waiter] 目标时刻: $(date -d "@$TGT" '+%F %T')" >> "$LOG"

while :; do
  NOW=$(date +%s)
  [ "$NOW" -ge "$TGT" ] && break
  LEFT=$((TGT - NOW))
  if [ "$LEFT" -gt 300 ]; then sleep 300; else sleep "$LEFT"; fi
done

echo "[waiter] 到达目标时刻，开始取回 $(date '+%F %T')" >> "$LOG"
bash "$PULL"
echo "[waiter] 取回脚本退出码=$? 完成于 $(date '+%F %T')" >> "$LOG"
