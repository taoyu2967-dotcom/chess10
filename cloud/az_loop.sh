#!/usr/bin/env bash
# 第二条数据线：GPU 自对弈（与 FSF 教师线并行；用 GPU，不抢 CPU 数据生成）
# 分批产出 spNNN_* 到 training/data_az，保留最近 AZ_KEEP 批
B=/root/autodl-tmp/chess
export CHESS10_WEIGHTS=$B/server/weights_ov.bin
GAMES=${AZ_GAMES:-20}
SIMS=${AZ_SIMS:-250}
MOVEMS=${AZ_MOVEMS:-700}
KEEP=${AZ_KEEP:-5}
mkdir -p "$B/training/data_az" "$B/logs"
cd "$B" || exit 1

n=0
for f in "$B"/training/data_az/sp*_encs.f32; do
  [ -e "$f" ] || continue
  k=$(basename "$f" | sed -E 's/^sp0*([0-9]+)_encs\.f32$/\1/')
  [ -n "$k" ] && [ "$k" -gt "$n" ] 2>/dev/null && n=$k
done
echo "[az] 启动 $(date '+%F %T') GAMES=$GAMES SIMS=$SIMS MOVEMS=$MOVEMS 从 sp$n 继续 保留 $KEEP 批"

while true; do
  n=$((n+1)); tag=$(printf 'sp%03d' "$n")
  echo "=== [az $tag] $(date '+%F %T') 自对弈 $GAMES 局 ==="
  node server/selfplay_gen.js "$GAMES" "$SIMS" "$MOVEMS" "$B/training/data_az/$tag" > "$B/logs/az_$tag.log" 2>&1
  rc=$?
  echo "[az $tag] rc=$rc  $(grep -o 'SP OK.*' "$B/logs/az_$tag.log" | tail -1)"
  # 轮转：只留最近 KEEP 批
  ls -1t "$B"/training/data_az/sp*_encs.f32 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r f; do
    s=${f%_encs.f32}
    rm -f "${s}_encs.f32" "${s}_pis.f32" "${s}_zs.f32"
    echo "  清理 $(basename "$s")"
  done
  df -h "$B" | tail -1 | awk '{print "  磁盘可用: "$4}'
done
