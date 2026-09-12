#!/usr/bin/env bash
B=/root/autodl-tmp/chess
cd "$B" || exit 1

echo "=== 1) 备份官方二进制 + 安装 vnni512 为在用版本 ==="
[ -f "$B/fsf/fsf-lb-bmi2" ] && cp -f "$B/fsf/fsf-lb-bmi2" "$B/fsf/fsf-lb-bmi2.official"
cp -f "$B/build/fsf-vnni512" "$B/fsf/fsf-lb-vnni512"
chmod +x "$B/fsf/fsf-lb-vnni512"
ln -sf fsf-lb-vnni512 "$B/fsf/fairy-stockfish"
ls -la "$B/fsf/" | grep -E "fairy-stockfish|fsf-lb"

echo "=== 2) 版本与变体自检 ==="
printf 'uci\nquit\n' | "$B/fsf/fairy-stockfish" 2>&1 | head -1
cd "$B/fsf" || exit 1
printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nposition startpos\ngo depth 6\nquit\n' \
  | ./fairy-stockfish 2>&1 | tail -2

echo "=== 3) 重启训练循环（追加日志，保留历史）==="
cd "$B" || exit 1
setsid bash loop.sh >> "$B/logs/loop.log" 2>&1 < /dev/null &
sleep 6
tail -4 "$B/logs/loop.log"
echo "FSF 进程数: $(pgrep -c fairy-stockfish 2>/dev/null || echo 0)"
echo "=== 结束 $(date +%H:%M:%S) ==="
