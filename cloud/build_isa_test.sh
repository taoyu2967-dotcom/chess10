#!/usr/bin/env bash
B=/root/autodl-tmp/chess
SRC=$B/src/Fairy-Stockfish-fairy_sf_14/src
mkdir -p "$B/build"
cd "$SRC" || exit 1
if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi

for ARCH in x86-64-bmi2 x86-64-avx512; do
  echo "=== 编译 $ARCH（nice -19 -j4）==="
  nice -n 19 make -j4 build ARCH=$ARCH largeboards=yes > /tmp/build_$ARCH.log 2>&1
  rc=$?
  tail -2 /tmp/build_$ARCH.log
  if [ $rc -ne 0 ]; then echo "  $ARCH 构建失败 rc=$rc"; make clean >/dev/null 2>&1; continue; fi
  found=0
  for f in fairy-stockfish*; do
    case "$f" in *.o|*.d|*.*) continue;; esac
    if [ -x "$f" ]; then cp -f "$f" "$B/build/fsf-$ARCH"; echo "  -> $B/build/fsf-$ARCH"; found=1; fi
  done
  [ $found -eq 0 ] && echo "  （未找到产物）"
  make clean >/dev/null 2>&1
done
ls -la "$B/build"

echo "=== 三版本 bench 对比（chess10d, 单线程 depth13）==="
cd "$B/fsf" || exit 1
for BIN in "$B/fsf/fsf-lb-bmi2" "$B/build/fsf-x86-64-bmi2" "$B/build/fsf-x86-64-avx512"; do
  [ -x "$BIN" ] || { echo "跳过（不存在） $BIN"; continue; }
  N=$(printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 1 13\nquit\n' | "$BIN" 2>&1 | grep -oE "Nodes/second +: [0-9]+" | grep -oE "[0-9]+")
  T=$(printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 1 13\nquit\n' | "$BIN" 2>&1 | grep -oE "Total time \(ms\) : [0-9]+" | grep -oE "[0-9]+")
  echo "$(basename "$BIN")  NPS=$N  用时=${T}ms"
done
echo "=== 结束 ==="
