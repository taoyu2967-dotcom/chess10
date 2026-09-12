#!/usr/bin/env bash
B=/root/autodl-tmp/chess
SRC=$B/src/Fairy-Stockfish-fairy_sf_14/src
mkdir -p "$B/build" "$B/logs"

echo "=== 1) 暂停训练循环（只杀我们自己的进程）==="
pkill -f "bash loop.sh" 2>/dev/null
pkill -f "selfplay_fsf_teacher" 2>/dev/null
pkill -f "fairy-stockfish" 2>/dev/null
sleep 3
echo "剩余相关进程数: $(pgrep -c -f 'loop\.sh|selfplay_fsf|fairy-stockfish' 2>/dev/null || echo 0)"

cd "$SRC" || exit 1
if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi

build_one() {  # $1=ARCH $2=target $3=outname
  echo "=== 编译 $3 [$2 ARCH=$1] $(date +%H:%M:%S) ==="
  make -j12 "$2" ARCH="$1" largeboards=yes > "$B/logs/mk_$3.log" 2>&1
  rc=$?
  if [ -x ./stockfish ]; then
    cp -f ./stockfish "$B/build/$3"
    echo "  OK -> $B/build/$3  $(stat -c%s "$B/build/$3")B  $(date +%H:%M:%S)"
  else
    echo "  FAIL rc=$rc 日志尾:"; tail -6 "$B/logs/mk_$3.log"
  fi
  make clean >/dev/null 2>&1
}

build_one x86-64-avx512  build         fsf-avx512
build_one x86-64-vnni512 build         fsf-vnni512
build_one x86-64-avx512  profile-build fsf-avx512-pgo

echo "=== 产物 ==="
ls -la "$B/build"
echo "=== 构建阶段结束 $(date +%H:%M:%S) ==="
