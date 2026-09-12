#!/usr/bin/env bash
set -e
B=/root/autodl-tmp/chess
cd "$B"
if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi

echo "=== 下载 FSF v14 源码 ==="
mkdir -p src && cd src
if [ ! -d Fairy-Stockfish-fairy_sf_14 ]; then
  timeout 600 curl -fL -o fsf14.tar.gz \
    "https://codeload.github.com/fairy-stockfish/Fairy-Stockfish/tar.gz/refs/tags/fairy_sf_14" \
    && tar xzf fsf14.tar.gz && rm -f fsf14.tar.gz
fi
ls -d Fairy-Stockfish-fairy_sf_14

echo "=== Makefile 支持的 x86-64 构建目标 ==="
grep -nE "x86-64[a-z0-9_-]*\)" Fairy-Stockfish-fairy_sf_14/src/Makefile | head -20
echo "=== largeboards 选项 ==="
grep -n "largeboards" Fairy-Stockfish-fairy_sf_14/src/Makefile | head -6
echo "=== 编译器 ==="
gcc --version | head -1
