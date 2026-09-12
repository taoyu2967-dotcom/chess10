#!/usr/bin/env bash
# 自对弈批参数扫描：测 positions/s 与 GPU 利用率，找更省时的批配置
B=/root/autodl-tmp/chess
cd "$B" || exit 1
export CHESS10_WEIGHTS=$B/server/weights_ov.bin
mkdir -p "$B/logs" "$B/training/data_az"

echo "=== 暂停 az_loop（不影响 FSF 训练线）==="
pkill -f "bash az_loop.sh" 2>/dev/null
pkill -f "selfplay_gen.js" 2>/dev/null
sleep 2
echo "剩余自对弈进程: $(pgrep -c -f selfplay_gen.js 2>/dev/null || echo 0)"

run_cfg() {  # $1=batch $2=flush $3=games $4=tag
  local b=$1 f=$2 g=$3 tag=$4
  echo "--- 配置 batchSize=$b flushMs=$f  games=$g ---"
  local t0=$(date +%s)
  CHESS10_AZ_BATCH=$b CHESS10_AZ_FLUSH=$f node server/selfplay_gen.js "$g" 250 700 "$B/training/data_az/sweep_$tag" > "$B/logs/sweep_$tag.log" 2>&1 &
  local pid=$!
  local peak=0
  for i in $(seq 1 40); do
    sleep 5
    kill -0 $pid 2>/dev/null || break
    local u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    [ "$u" -gt "$peak" ] 2>/dev/null && peak=$u
  done
  wait $pid
  local t1=$(date +%s)
  local pos=$(grep -o 'positions=[0-9]*' "$B/logs/sweep_$tag.log" | tail -1 | grep -o '[0-9]*')
  local el=$((t1-t0))
  echo "  ${el}s  positions=${pos:-0}  峰值GPU=${peak}%  → 吞吐 $(awk "BEGIN{printf \"%.2f\", ${pos:-0}/$el}") 局面/秒"
}

run_cfg 64  200 4 base
run_cfg 32  50  4 b32f50
run_cfg 128 30  4 b128f30

echo "=== 清理扫描产物 ==="
rm -f "$B/training/data_az/sweep_"*_encs.f32 "$B/training/data_az/sweep_"*_pis.f32 "$B/training/data_az/sweep_"*_zs.f32
echo "=== 恢复 az_loop ==="
setsid bash az_loop.sh >> "$B/logs/az_loop.log" 2>&1 < /dev/null &
sleep 5
echo "az_loop 进程: $(pgrep -f 'bash az_loop.sh' >/dev/null && echo OK || echo FAIL)"
echo "=== 结束 $(date +%H:%M:%S) ==="
