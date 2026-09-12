#!/usr/bin/env bash
# chess10 云端训练循环：FSF 教师生成 → CUDA 训练 → 核验 → 转正 → 数据轮转
# 断线不中断：用 setsid + nohup 启动
B=/root/autodl-tmp/chess
export CHESS10_TEACHER=$B/training/teacher
export CHESS10_AZOV=$B/training/data
export CHESS10_FSF=$B/fsf/fairy-stockfish
export CHESS10_FSF_CORES=${CHESS10_FSF_CORES:-0,1,2,3,4,5,6,7,8,9,10,11}
export CHESS10_W_IN=$B/server/weights_ov.bin
export CHESS10_W_OUT=$B/server/weights_ov_new.bin
# 第二条线：GPU 自对弈数据（默认 2000 上限；设为 0 即关闭这条线的数据）
export CHESS10_AZ_DIR=$B/training/data_az
export CHESS10_AZ_CAP=${CHESS10_AZ_CAP:-2000}
PY=/root/miniconda3/bin/python
GAMES=${GAMES:-120}
MT=${MT:-150}
EPS=${EPS:-4}
KEEP=${KEEP:-3}

mkdir -p "$B/logs" "$B/snapshots" "$B/training/data"
cd "$B" || exit 1
echo "[loop] 启动 $(date '+%F %T')  GAMES=$GAMES MT=$MT EPS=$EPS 核=$CHESS10_FSF_CORES"

r=0
for d in "$B"/training/data/r*_encs.f32; do
  [ -e "$d" ] || continue
  n=$(basename "$d" | sed -E 's/^r0*([0-9]+)_encs\.f32$/\1/')
  [ -n "$n" ] && [ "$n" -gt "$r" ] 2>/dev/null && r=$n
done
echo "[loop] 已有最大轮次 r$r"

while true; do
  r=$((r+1))
  tag=$(printf 'r%03d' "$r")
  echo "=== [$tag] $(date '+%F %T') 教师生成 $GAMES 局 ==="
  node server/selfplay_fsf_teacher.js "$GAMES" "$MT" "$B/training/data/$tag" > "$B/logs/sp_$tag.log" 2>&1
  if [ $? -ne 0 ]; then echo "[$tag] 教师生成失败，跳过本轮"; sleep 20; continue; fi
  echo "[$tag] $(grep -o 'FSF TEACHER OK.*' "$B/logs/sp_$tag.log" | tail -1)"

  echo "=== [$tag] $(date '+%F %T') 训练 $EPS epoch ==="
  "$PY" ov_train/torch_ov_train.py "$EPS" 1e-4 "$r" > "$B/logs/tr_$tag.log" 2>&1
  if [ $? -ne 0 ]; then echo "[$tag] 训练/门禁失败，保留旧权重"; tail -2 "$B/logs/tr_$tag.log"; continue; fi
  grep -E "pre-train probe|post-train probe|exported" "$B/logs/tr_$tag.log" | tr '\n' ' '; echo

  if node server/verify_weights.js "$B/server/weights_ov_new.bin" > "$B/logs/vf_$tag.log" 2>&1; then
    cp -f "$B/server/weights_ov_new.bin" "$B/server/weights_ov.bin"
    cp -f "$B/server/weights_ov.bin" "$B/snapshots/$tag.bin"
    echo "[$tag] 转正 OK：$(cat "$B/logs/vf_$tag.log")"
  else
    echo "[$tag] VERIFY 失败，保留旧权重"
  fi

  # 数据轮转：只保留最近 KEEP 轮（每轮约 1.6GB）
  ls -1t "$B"/training/data/r*_encs.f32 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r f; do
    s=${f%_encs.f32}
    rm -f "${s}_encs.f32" "${s}_pis.f32" "${s}_zs.f32"
    echo "  清理 $(basename "$s")"
  done
  echo "[$tag] 完成 $(date '+%F %T')  磁盘: $(df -h "$B" | awk 'NR==2{print $4" 可用"}')"
done
