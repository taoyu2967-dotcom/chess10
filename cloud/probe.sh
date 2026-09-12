#!/usr/bin/env bash
# 云端环境体检（通过 ssh 'bash -s' 管道执行）
echo "=== SSH OK ==="
hostname; uname -srm
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用"
echo "=== CPU / MEM ==="
echo "cores=$(nproc)"
free -g | sed -n '1,2p'
echo "=== DISK ==="
df -h / /root /root/autodl-tmp 2>/dev/null | awk 'NR==1 || !seen[$6]++'
echo "=== PYTHON / TORCH ==="
python3 -V
python3 - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| available", torch.cuda.is_available(),
          "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu"))
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("torch/numpy 检查失败:", repr(e))
PY
echo "=== 工具链 ==="
for t in node npm git gcc g++ make cmake unzip wget curl; do
  printf "%s=%s " "$t" "$(command -v "$t" >/dev/null 2>&1 && echo yes || echo NO)"
done; echo
echo "=== 网络（镜像源可达性）==="
timeout 8 curl -sI https://mirrors.aliyun.com 2>/dev/null | head -1 || echo "aliyun 镜像不通"
timeout 8 curl -sI https://github.com 2>/dev/null | head -1 || echo "github 不通"
echo "=== 结束 ==="
