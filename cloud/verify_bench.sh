#!/usr/bin/env bash
B=/root/autodl-tmp/chess
cd "$B/fsf" || exit 1
[ -f variants.ini ] || { echo "缺 variants.ini"; exit 1; }

run_bench() {  # $1=bin  $2=threads
  local out
  out=$(printf 'setoption name VariantPath value variants.ini\nsetoption name UCI_Variant value chess10d\nbench 16 %s 13\nquit\n' "$2" | "$1" 2>&1)
  local n t
  n=$(echo "$out" | grep -oE "Nodes searched +: [0-9]+" | grep -oE "[0-9]+$")
  t=$(echo "$out" | grep -oE "Nodes/second +: [0-9]+" | grep -oE "[0-9]+$")
  echo "$n $t"
}

echo "############ A2 等价性验证：Nodes searched 必须与官方一致 ############"
echo "（chess10d, bench 16 1 13）"
declare -A BINS=(
  [official-bmi2]="$B/fsf/fsf-lb-bmi2"
  [my-bmi2]="$B/build/fsf-x86-64-bmi2"
  [my-avx512]="$B/build/fsf-avx512"
  [my-vnni512]="$B/build/fsf-vnni512"
  [my-avx512-pgo]="$B/build/fsf-avx512-pgo"
)
for k in official-bmi2 my-bmi2 my-avx512 my-vnni512 my-avx512-pgo; do
  bin=${BINS[$k]}
  [ -x "$bin" ] || { printf "%-16s 缺失\n" "$k"; continue; }
  read -r n t < <(run_bench "$bin" 1)
  printf "%-16s Nodes=%-8s NPS=%s\n" "$k" "$n" "$t"
done

echo
echo "############ A3 干净对比：单线程 3 轮取中位数 ############"
for k in official-bmi2 my-avx512 my-vnni512 my-avx512-pgo; do
  bin=${BINS[$k]}
  [ -x "$bin" ] || continue
  vals=""
  for i in 1 2 3; do
    read -r n t < <(run_bench "$bin" 1)
    vals="$vals $t"
  done
  med=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | sed -n '2p')
  printf "%-16s 单线程 NPS:%s  中位数=%s\n" "$k" "$vals" "$med"
done

echo
echo "############ A3 干净对比：12 线程 3 轮取中位数 ############"
for k in official-bmi2 my-avx512 my-avx512-pgo; do
  bin=${BINS[$k]}
  [ -x "$bin" ] || continue
  vals=""
  for i in 1 2 3; do
    read -r n t < <(run_bench "$bin" 12)
    vals="$vals $t"
  done
  med=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | sed -n '2p')
  printf "%-16s 12线程 NPS:%s  中位数=%s\n" "$k" "$vals" "$med"
done
echo "=== 结束 $(date +%H:%M:%S) ==="
