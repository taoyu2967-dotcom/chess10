#!/usr/bin/env bash
B=/root/autodl-tmp/chess
cd "$B" || exit 1
export CHESS10_WEIGHTS=$B/server/weights_ov.bin
mkdir -p "$B/training/data_az" "$B/logs"
echo "=== 自对弈冒烟：2 局 / 250 sims / 700ms 上限 ==="
echo "权重: $(ls -l $CHESS10_WEIGHTS | awk '{print $5}') bytes"
start=$(date +%s)
node server/selfplay_gen.js 2 250 700 "$B/training/data_az/sp000" 2>&1 | tail -12
end=$(date +%s)
echo "耗时: $((end-start)) 秒 / 2 局"

echo "=== 产物与宽度核对 ==="
ls -l "$B/training/data_az/" | awk '{print $5, $9}'
/root/miniconda3/bin/python - <<'PY'
import os
d="/root/autodl-tmp/chess/training/data_az/"
try:
    e=os.path.getsize(d+"sp000_encs.f32")//4//2400
    p=os.path.getsize(d+"sp000_pis.f32")//4//16000
    z=os.path.getsize(d+"sp000_zs.f32")//4
    print(f"局面数 encs={e} pis(16000宽)={p} zs={z} 一致={'OK' if e==p==z else 'FAIL'}")
    if e>0:
        import time
        print(f"吞吐参考: 见上面耗时 / {e} 局面")
except Exception as ex:
    print("核对异常:", ex)
PY
echo "=== 结束 $(date +%H:%M:%S) ==="
