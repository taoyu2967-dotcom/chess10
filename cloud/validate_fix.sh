#!/usr/bin/env bash
B=/root/autodl-tmp/chess
cd "$B" || exit 1

echo "=== 1) 暂停训练循环（不动 az_loop）==="
pkill -f "bash loop.sh" 2>/dev/null
pkill -f "selfplay_fsf_teacher" 2>/dev/null
pkill -f "fairy-stockfish" 2>/dev/null
sleep 3
echo "剩余 FSF 进程: $(pgrep -c fairy-stockfish 2>/dev/null || echo 0)"
echo "az_loop 是否仍在: $(pgrep -f 'bash az_loop.sh' >/dev/null && echo 是 || echo 否)"

echo "=== 2) 计时验证：12 局 / 12 引擎（修复后）==="
export CHESS10_FSF=$B/fsf/fairy-stockfish
export CHESS10_FSF_CORES=0,1,2,3,4,5,6,7,8,9,10,11
start=$(date +%s)
node server/selfplay_fsf_teacher.js 12 150 "$B/training/data/val_fix" > "$B/logs/val_fix.log" 2>&1
rc=$?
end=$(date +%s)
echo "rc=$rc  实际耗时 $((end-start)) 秒"
grep -o "FSF TEACHER OK.*" "$B/logs/val_fix.log" | tail -1
echo "兜底次数（应为 0 或极少）: $(grep -o '兜底[0-9]*' "$B/logs/val_fix.log" | tail -1)"
echo
echo "对照：按 r001 实测（120 局 / 1455 秒）线性折算，12 局旧实现约需 $((1455*12/120)) 秒"
echo "=== 结束 $(date +%H:%M:%S) ==="
