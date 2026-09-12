#!/usr/bin/env bash
B=/root/autodl-tmp/chess
echo "=== loop.log 全文（尾 12 行）==="
tail -12 "$B/logs/loop.log" 2>/dev/null || echo "(无 loop.log)"
echo "=== sp_r001.log 尾部 ==="
tail -8 "$B/logs/sp_r001.log" 2>/dev/null || echo "(无 sp_r001.log)"
echo "=== 进程（按 CPU 排序前 8）==="
ps -eo comm,pcpu,etime --sort=-pcpu 2>/dev/null | head -9
echo -n "node 进程数: "; pgrep -c node 2>/dev/null || echo 0
echo -n "fsf  进程数: "; pgrep -c fsf 2>/dev/null || echo 0
echo "=== 数据目录 ==="
ls -la "$B/training/data/" 2>/dev/null | tail -6

cd "$B/fsf/src/src" || exit 1
echo "=== 保存 avx512 构建 ==="
cp -f stockfish "$B/fsf/fsf-avx512" && ls -l "$B/fsf/fsf-avx512" | awk '{print $5, $9}'
echo "=== clean 后编译 avx2 对照组 ==="
make clean >/dev/null 2>&1
make -j6 build ARCH=x86-64-avx2 largeboards=yes 2>&1 | tail -2
cp -f stockfish "$B/fsf/fsf-avx2" && ls -l "$B/fsf/fsf-avx2" | awk '{print $5, $9}'
echo "=== 待测三件套 ==="
for f in fsf-lb-bmi2 fsf-avx2 fsf-avx512; do
  [ -x "$B/fsf/$f" ] && echo "OK $f"
done
echo "=== 结束 ==="
