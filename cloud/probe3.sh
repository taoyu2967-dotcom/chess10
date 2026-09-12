#!/usr/bin/env bash
echo "=== 学术加速 ==="
if [ -f /etc/network_turbo ]; then
  echo "/etc/network_turbo 存在"
  # shellcheck disable=SC1091
  source /etc/network_turbo && echo "已启用"
  env | grep -i proxy | head -3
else
  echo "无 /etc/network_turbo"
fi
for u in https://github.com https://raw.githubusercontent.com https://codeload.github.com; do
  printf "%-40s " "$u"
  timeout 12 curl -sI -o /dev/null -w "%{http_code}\n" "$u" 2>/dev/null || echo FAIL
done
echo "=== torch / numpy ==="
/root/miniconda3/bin/python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available(), "|",
      (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu"))
print("numpy", numpy.__version__)
print("arch_list 含 sm_75:", "sm_75" in torch.cuda.get_arch_list())
PY
echo "=== 结束 ==="
