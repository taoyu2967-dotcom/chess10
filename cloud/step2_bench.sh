#!/usr/bin/env bash
cd /root/autodl-tmp/chess/fsf || exit 1
echo "=== variants.ini ==="
wc -l variants.ini 2>/dev/null || { echo "缺 variants.ini"; exit 1; }
echo "=== chess10d 变体自检（开局搜 6 层）==="
printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nposition startpos\ngo depth 6\nquit\n' \
  | ./fairy-stockfish 2>&1 | tail -3
echo "=== bench 单线程 depth13（与笔记本 600k NPS 对比）==="
printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 1 13\nquit\n' \
  | ./fairy-stockfish 2>&1 | tail -5
echo "=== bench 12 线程 depth13（并行总吞吐）==="
printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 12 13\nquit\n' \
  | ./fairy-stockfish 2>&1 | tail -4
echo "=== 结束 ==="
