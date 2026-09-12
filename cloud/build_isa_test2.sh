#!/usr/bin/env bash
B=/root/autodl-tmp/chess
SRC=$B/src/Fairy-Stockfish-fairy_sf_14/src
mkdir -p "$B/build"
cd "$SRC" || exit 1
if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi

for ARCH in x86-64-bmi2 x86-64-avx512; do
  echo "=== 编译 $ARCH ==="
  nice -n 19 make -j4 build ARCH=$ARCH largeboards=yes > /tmp/build_$ARCH.log 2>&1
  if [ -x ./stockfish ]; then
    cp -f ./stockfish "$B/build/fsf-$ARCH"
    echo "  -> $B/build/fsf-$ARCH  $(stat -c%s "$B/build/fsf-$ARCH") bytes"
  else
    echo "  $ARCH 未产出，尾部日志："; tail -3 /tmp/build_$ARCH.log
  fi
  make clean >/dev/null 2>&1
done
ls -la "$B/build"

echo "=== bench 对比（chess10d，单线程 depth13，跑两轮取稳定值）==="
cd "$B/fsf" || exit 1
for BIN in "$B/fsf/fsf-lb-bmi2" "$B/build/fsf-x86-64-bmi2" "$B/build/fsf-x86-64-avx512"; do
  [ -x "$BIN" ] || { echo "跳过（不存在） $(basename "$BIN")"; continue; }
  for round in 1 2; do
    OUT=$(printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 1 13\nquit\n' | "$BIN" 2>&1)
    N=$(echo "$OUT" | grep -oE "Nodes/second +: [0-9]+" | grep -oE "[0-9]+")
    T=$(echo "$OUT" | grep -oE "Total time \(ms\) : [0-9]+" | grep -oE "[0-9]+")
    echo "$(basename "$BIN")  round$round  NPS=$N  用时=${T}ms"
  done
done
echo "=== 12 线程对照（并行总吞吐）==="
for BIN in "$B/fsf/fsf-lb-bmi2" "$B/build/fsf-x86-64-avx512"; do
  [ -x "$BIN" ] || continue
  OUT=$(printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 12 13\nquit\n' | "$BIN" 2>&1)
  N=$(echo "$OUT" | grep -oE "Nodes/second +: [0-9]+" | grep -oE "[0-9]+")
  echo "$(basename "$BIN")  12线程 NPS=$N"
done
echo "=== 结束 ==="
