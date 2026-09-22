#!/usr/bin/env bash
# 从 AutoDL 实例取回 chess10 训练产物（幂等：按文件大小跳过已下载；可重复执行）
# 取回内容：关键权重 + 全部快照 + 关键日志（不取 training/data，那是可再生的且很大）
set -u
SSHOPT="-i $HOME/.ssh/autodl_chess10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20"
R=root@region-42.seetacloud.com
P=47010
RB=/root/autodl-tmp/chess
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cloud_pull"   # 路径中枢约定：脚本位置推导
LOG="$DEST/pull.log"

mkdir -p "$DEST/snapshots" "$DEST/server" "$DEST/logs"
exec >> "$LOG" 2>&1

echo "==================== $(date '+%F %T') 取回开始 ===================="

# 互斥锁（陈旧锁 >3 小时自动清理）
LOCK="$DEST/.lock"
if [ -d "$LOCK" ]; then find "$LOCK" -maxdepth 0 -mmin +180 -exec rmdir {} \; 2>/dev/null; fi
if ! mkdir "$LOCK" 2>/dev/null; then echo "SKIP: 另一个取回任务正在运行"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# 连通性与实例存活
if ! ssh $SSHOPT -p "$P" "$R" 'echo ok' >/dev/null 2>&1; then
  echo "FAIL: 无法连接实例（可能已到期/关机）"; exit 1
fi

echo "--- 1) 关键权重（带 sha256 校验）---"
for f in server/weights_ov.bin server/weights_ov_new.bin; do
  # 权重每轮都变但尺寸恒为 13,200,980 字节，按大小跳过会永远漏更新，必须每次重下（13MB 很便宜）
  scp $SSHOPT -P "$P" "$R:$RB/$f" "$DEST/$f" >/dev/null 2>&1 \
    || { echo "  FAIL $f（云端不存在或传输失败，保留本地旧版）"; continue; }
  echo "  取回 $(basename "$f") $(stat -c %s "$DEST/$f") bytes"
  rsha=$(ssh $SSHOPT -p "$P" "$R" "sha256sum $RB/$f | cut -d' ' -f1")
  lsha=$(sha256sum "$DEST/$f" | cut -d' ' -f1)
  if [ "$rsha" = "$lsha" ]; then echo "  sha256 一致 ✓ $lsha"; else echo "  ⚠ sha256 不一致！remote=$rsha local=$lsha"; fi
done

echo "--- 2) 快照（按大小跳过已下载）---"
ssh $SSHOPT -p "$P" "$R" "cd $RB/snapshots 2>/dev/null && ls -l --time-style=+ *.bin | awk '{print \$5\" \"\$6}'" > "$DEST/.remote_snaps" 2>/dev/null || true
exists=0; new=0; fails=0
while read -r sz nm; do
  [ -z "${nm:-}" ] && continue
  LS=$(stat -c %s "$DEST/snapshots/$nm" 2>/dev/null || echo 0)
  if [ "$LS" = "$sz" ]; then exists=$((exists+1)); continue; fi
  if scp $SSHOPT -P "$P" "$R:$RB/snapshots/$nm" "$DEST/snapshots/$nm" >/dev/null 2>&1; then new=$((new+1)); else fails=$((fails+1)); echo "  FAIL $nm"; fi
done < "$DEST/.remote_snaps"
echo "  已存在 $exists 个，本次新取 $new 个，失败 $fails 个"
echo "  本地快照总数: $(ls "$DEST/snapshots"/*.bin 2>/dev/null | wc -l)"

echo "--- 3) 关键日志 ---"
scp $SSHOPT -P "$P" "$R:$RB/logs/loop.log" "$DEST/logs/loop.log" >/dev/null 2>&1 && echo "  loop.log ✓" || echo "  loop.log FAIL"
ssh $SSHOPT -p "$P" "$R" "cd $RB/logs && tar czf - tr_*.log vf_*.log az_loop.log 2>/dev/null" > "$DEST/logs/detail_logs.tgz" 2>/dev/null \
  && echo "  detail_logs.tgz $(stat -c %s "$DEST/logs/detail_logs.tgz") bytes" || echo "  日志打包 FAIL"

echo "--- 4) 汇总 ---"
echo "  远端快照数: $(wc -l < "$DEST/.remote_snaps" 2>/dev/null || echo '?')"
echo "  本地快照数: $(ls "$DEST/snapshots"/*.bin 2>/dev/null | wc -l)"
echo "  目录大小:   $(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "==================== $(date '+%F %T') 取回结束 ===================="
