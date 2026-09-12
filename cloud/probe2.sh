#!/usr/bin/env bash
echo "=== conda/python 定位 ==="
for p in /root/miniconda3/bin/python /opt/conda/bin/python /usr/bin/python3 /usr/local/bin/python3; do
  [ -x "$p" ] && echo "OK $p -> $("$p" -V 2>&1)"
done
echo "=== 真实资源配额（cgroup）==="
echo -n "cpu.max: "; cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null
echo -n "mem.max: "; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
echo "=== pip / conda ==="
ls /root/miniconda3/bin/pip /root/miniconda3/bin/conda 2>/dev/null
echo "=== github 与加速源可达性 ==="
for u in https://github.com https://raw.githubusercontent.com https://mirrors.aliyun.com https://mirrors.tuna.tsinghua.edu.cn; do
  printf "%-45s " "$u"
  timeout 10 curl -sI -o /dev/null -w "%{http_code}\n" "$u" 2>/dev/null || echo FAIL
done
echo "=== AutoDL 学术加速是否已配置 ==="
grep -c . /etc/hosts 2>/dev/null | head -1
grep -i "github\|academic" /etc/hosts 2>/dev/null | head -3 || echo "(hosts 无 github 条目)"
echo "=== 结束 ==="
