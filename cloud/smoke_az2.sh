#!/usr/bin/env bash
B=/root/autodl-tmp/chess
cd "$B" || exit 1
export CHESS10_WEIGHTS=$B/server/weights_ov.bin
mkdir -p "$B/training/data_az" "$B/logs"

echo "=== 自对弈冒烟（修复 GPU 权重上传后）2 局 / 250 sims / 700ms ==="
echo "权重: $CHESS10_WEIGHTS $(stat -c%s "$CHESS10_WEIGHTS") bytes"
start=$(date +%s)
node server/selfplay_gen.js 2 250 700 "$B/training/data_az/sp001" > "$B/logs/az_sp001.log" 2>&1
rc=$?
end=$(date +%s)
echo "rc=$rc  耗时 $((end-start)) 秒"
echo "--- 自对弈 stdout 尾部 ---"
tail -10 "$B/logs/az_sp001.log"
echo "--- 产物 ---"
ls -l "$B/training/data_az/" | awk '{print $5, $9}'
/root/miniconda3/bin/python - <<'PY'
import os
d="/root/autodl-tmp/chess/training/data_az/"
try:
    e=os.path.getsize(d+"sp001_encs.f32")//4//2400
    p=os.path.getsize(d+"sp001_pis.f32")//4//16000
    z=os.path.getsize(d+"sp001_zs.f32")//4
    print(f"局面数 encs={e} pis(16000宽)={p} zs={z} 一致={'OK' if e==p==z else 'FAIL'}")
except Exception as ex:
    print("核对异常:", ex)
PY
echo "=== 结束 $(date +%H:%M:%S) ==="
