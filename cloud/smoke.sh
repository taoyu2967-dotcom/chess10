#!/usr/bin/env bash
set -e
B=/root/autodl-tmp/chess
cd "$B"
export CHESS10_TEACHER=$B/training/teacher
export CHESS10_AZOV=$B/training/data
export CHESS10_W_IN=$B/server/weights_ov.bin
export CHESS10_W_OUT=$B/server/weights_ov_new.bin
PY=/root/miniconda3/bin/python

echo "=== 1) 重生教师数据（16000 宽）==="
node server/export_for_pytorch.js 3000 60 2>&1 | tail -4
echo "--- 宽度核对 ---"
ls -l $B/training/teacher/train_encs.f32 $B/training/teacher/train_pis.f32 $B/training/teacher/train_zs.f32 \
  | awk '{print $5, $9}'
$PY - <<'PY'
import os
t="/root/autodl-tmp/chess/training/teacher/"
import numpy as np
e=os.path.getsize(t+"train_encs.f32")//4//2400
p=os.path.getsize(t+"train_pis.f32")//4//16000
print(f"局面数 encs={e} pis(16000宽)={p} 一致={'OK' if e==p else 'FAIL'}")
PY

echo "=== 2) 1 epoch 训练（Arm 0）==="
$PY ov_train/torch_ov_train.py 1 1e-4 0 2>&1 | tail -8

echo "=== 3) 权重核验 ==="
node server/verify_weights.js $B/server/weights_ov_new.bin
ls -l $B/server/weights_ov_new.bin | awk '{print $5, $9, "floats:", $5/4}'
echo "=== 结束 ==="
