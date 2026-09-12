#!/usr/bin/env bash
B=/root/autodl-tmp/chess
echo "=== 1) 补 OpenCL ICD 文件（唯一缺环）==="
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
cat /etc/OpenCL/vendors/nvidia.icd

echo "=== 2) 装 opencl-raub（npmmirror 源）==="
cd "$B/server" || exit 1
npm config set registry https://registry.npmmirror.com >/dev/null 2>&1
timeout 600 npm install opencl-raub --no-audit --no-fund 2>&1 | tail -6
if [ -d node_modules/opencl-raub ]; then echo "opencl-raub 已安装"; else echo "opencl-raub 安装失败"; fi

echo "=== 3) OpenCL 设备枚举 ==="
cp -f "$B/ocltest.js" "$B/server/ocltest.js" 2>/dev/null
node "$B/server/ocltest.js" 2>&1 | tail -6

echo "=== 4) 候选编译进度 ==="
tail -4 "$B/logs/build_all.log"
