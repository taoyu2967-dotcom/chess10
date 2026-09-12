#!/usr/bin/env bash
cd /root || exit 1
echo "=== node 检查 ==="
if command -v node >/dev/null 2>&1; then echo "node $(node -v) / npm $(npm -v)"; else echo "NODE_MISSING"; fi
mkdir -p /root/autodl-tmp/chess/fsf
cd /root/autodl-tmp/chess/fsf || exit 1
echo "=== 下载 FSF 14 largeboard bmi2（官方预编译，走学术加速）==="
if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi
if [ ! -x ./fsf-lb-bmi2 ]; then
  timeout 300 curl -fL -o fsf-lb-bmi2 \
    "https://github.com/fairy-stockfish/Fairy-Stockfish/releases/download/fairy_sf_14/fairy-stockfish-largeboard_x86-64-bmi2" \
    && chmod +x fsf-lb-bmi2 && ln -sf fsf-lb-bmi2 fairy-stockfish
fi
ls -la
echo "=== 版本自报 ==="
printf 'uci\nquit\n' | ./fairy-stockfish 2>&1 | head -3
echo "=== 结束 ==="
