#!/usr/bin/env bash
cd /root || exit 1

echo "=== 1) 安装 node（清华镜像，不走代理）==="
if command -v node >/dev/null 2>&1; then
  echo "node 已存在: $(node -v)"
else
  ok=0
  for V in v20.18.0 v20.11.1 v18.20.4; do
    url="https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/${V}/node-${V}-linux-x64.tar.xz"
    echo "尝试 $url"
    if timeout 300 curl -fsSL -o /root/node.tar.xz "$url"; then ok=1; NODE=$V; break; fi
  done
  if [ "$ok" = "1" ]; then
    tar -xJf /root/node.tar.xz -C /root
    mkdir -p /usr/local/node
    cp -r "/root/node-${NODE}-linux-x64/." /usr/local/node/
    ln -sf /usr/local/node/bin/node /usr/local/bin/node
    ln -sf /usr/local/node/bin/npm /usr/local/bin/npm
    ln -sf /usr/local/node/bin/npx /usr/local/bin/npx
    echo "node 安装完成: $(node -v) / npm $(npm -v)"
  else
    echo "node 下载失败"
  fi
fi

echo "=== 2) FSF GitHub release 资产（走学术加速）==="
if [ -f /etc/network_turbo ]; then . /etc/network_turbo >/dev/null 2>&1; fi
timeout 90 curl -s "https://api.github.com/repos/fairy-stockfish/Fairy-Stockfish/releases?per_page=6" \
  | grep -E '"tag_name"|"name":|browser_download_url' | head -50
echo "=== 结束 ==="
