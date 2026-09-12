#!/usr/bin/env bash
B=/root/autodl-tmp/chess
echo "=== 当前负载 ==="
echo -n "FSF 进程数: "; pgrep -c fsf 2>/dev/null || echo 0
echo -n "训练进程数: "; pgrep -fc torch_ov_train 2>/dev/null || echo 0
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- loop.log 尾部 ---"
tail -4 "$B/logs/loop.log" 2>/dev/null || echo "(无 loop.log)"

cd "$B/fsf" || exit 1
echo "=== 下载 FSF 14 源码（走学术加速）==="
if [ ! -d src ]; then
  if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi
  timeout 300 curl -fL -o fsfsrc.tgz \
    "https://codeload.github.com/fairy-stockfish/Fairy-Stockfish/tar.gz/refs/tags/fairy_sf_14" \
    && mkdir -p src && tar xzf fsfsrc.tgz -C src --strip-components=1
fi
ls -l fsfsrc.tgz 2>/dev/null | awk '{print $5, $9}'
ls "$B/fsf/src/src/Makefile" 2>/dev/null && echo "Makefile OK"

echo "=== Makefile 支持的 x86-64 ARCH ==="
grep -oE "x86-64[a-z0-9-]*" "$B/fsf/src/src/Makefile" 2>/dev/null | sort -u | tr '\n' ' '
echo

echo "=== 编译 avx512（-j8，留几个核）==="
cd "$B/fsf/src/src" || exit 1
make -j8 build ARCH=x86-64-avx512 largeboards=yes 2>&1 | tail -6
ls -la fairy-stockfish-largeboard* 2>/dev/null | awk '{print $5, $9}'

echo "=== 编译 avx2（对照组）==="
make -j8 build ARCH=x86-64-avx2 largeboards=yes 2>&1 | tail -4
ls -la fairy-stockfish-largeboard* 2>/dev/null | awk '{print $5, $9}'
echo "=== 结束 ==="
