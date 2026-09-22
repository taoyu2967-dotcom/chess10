# chess10d 运维手册（RUNBOOK）

> 面向"接手的人"（包括未来的自己/AI）：怎么把这套东西跑起来、停下去、坏了怎么判。

## 快速启动

```bash
# 对弈（权重就绪即开）
cp weights/BJ1_r208_v3.bin server/weights_ov.bin   # 或任选 weights/ 下的权重
npm run serve                                        # = node server/server.js
# 浏览器打开 http://127.0.0.1:8787/chess10.html
# 引擎可选：MCTS（OpenCL GPU 池，默认）/ ⚡MCTS-NPU（OpenVINO，需 Intel NPU）

# 桌面壳（WinUI 3 + WebView2，自动拉起/回收 node 服务）
dotnet build winui/Chess10d.csproj -c Release -p:Platform=x64
winui\bin\x64\Release\net8.0-windows10.0.19041.0\Chess10d.exe
```

## 门禁套件（任何改动后跑）

```bash
npm test            # 全量（GPU/py/NPU 依赖项全跑，约 5-8 分钟）
npm run test:quick  # 纯 JS 核心门禁（无卡环境，约 1 分钟）
```

全量项：policy 编码单射 / 走法与攻击等价差分 / cnn v2+v3 对拍 / GPU v3 对拍 /
e2e 冒烟（真实起服+WS）/ 集成级门禁（mano patch）/ OpenVINO×v3（NPU）/ 权重探针×4。
唯一非阻塞项：GPU v2 对拍（已知历史失败，见 docs/contracts/v3_contract.md §6，WARN）。

## 训练（本地 Windows 守护）

```powershell
# 启动（幂等：已在跑则提示 pid 后退出）
powershell -NoProfile -ExecutionPolicy Bypass -File training/start_daemon.ps1
# 停止：创建旗标（守护当轮收尾后退出）
New-Item training/STOP.flag
# 日常收尾统计（巡检第 2 步用它）
node training/housekeeping.js
```

守护行为：FSF 教师生成 → CUDA 训练 → 探针+verify 双门禁 → 转正+快照；
睡眠窗口 14:00-17:59；GPU 自愈（restore_gpu_stack.js 每轮至多一次）；
数据轮换留最新 6 轮、快照留最新 40。

## 探针门禁语义（判读训练健康）

每轮训练前后跑固定三题（`server/verify_weights.js` 同款局面）：

- **初始局面**：应近 0（±0.1 量级）
- **白优局面**（9k 残局，白大优）：应在 0.15 门禁线以上；**跌破 0.15 = 本轮作废**
  长期缓慢下行属学习轨迹（曾 1.31→0.60），连续急跌要看数据配方（AZ 和棋标签占比）
- **中局局面**：应近 0

历史案例：r209/r210 白优探针 0.108/0.175 触发门禁连续拦截 → 判定为
AZ 混入 z=0 标签致 value 头漂移（未根治，恢复训练前先复核标签分布）。

## OpenVINO/NPU 通道

```bash
CHESS10_BACKEND=openvino node server/server.js        # NPU 后端
CHESS10_OV_DEVICE=GPU.0 ...                            # 换 Arc iGPU（快 3×）
node tests/test_ov_v3_weights.js <权重> NPU            # 单独验证某权重上 NPU
py -3 tests/ov_b1_diag.py                              # b1 档精度诊断
```

- 桥按批档位编译（b32/b16/b8/b1），每档独立过精度门禁才入列；**b1 档门禁是
  margin-aware 的**（参照并列处 argmax 掷硬币不算失败，2026-09-17 修正）
- fp16 精度门禁：value≤0.05 / policy≤3.0 / 非并列位置 top1 全一致

## 云端时代（AutoDL，已到期——历史档案）

cloud/*.sh 曾驱动北京A区 2080Ti 实例的无人值守循环（loop.sh=教师蒸馏线、
az_loop.sh=GPU 自对弈线，~6.5 分/轮）。实例 2026-09-12 到期，血统代号 **BJ-1**，
终版权重 r211 已取回本地（cloud_pull/BJ1/，含 169 快照与训练数据）。
SSH 配置（-p 47010 region-42.seetacloud.com）与 /root/autodl-tmp/chess 路径
均为该实例专用，重启新实例需按脚本内注释改。

## 故障速查

| 症状 | 判定 | 处置 |
|---|---|---|
| GPU 后端静默回退 CPU（GPU 0%） | mcts.loadWeights 的上传时序 | 已修（gpu.hasContext 路径）；复现先跑 gpu_v3 对拍 |
| OV 桥挂死无响应 | 用了旧协议客户端 ov_bridge_client.js（已移出） | 直连 ov_bridge_server_v2.py 简单协议 |
| NPU b1 档缺失（tiers 无 1） | 看 [OV] 日志 gate FAIL 行 | py -3 tests/ov_b1_diag.py 定位（b1_diag 输出 margin/Δ 判读表） |
| 训练轮全被门禁拦 | 白优探针跌破 0.15 | 看 RUNBOOK 探针节；先查数据配方再查代码 |
| opencl-raub 坏 | node -e "require('./server/gpu').init()" 报错 | node server/tools/restore_gpu_stack.js（自愈修复器） |

## 退役区说明

`_legacy/`（本地 gitignored）存放架构现代化时移出的：单文件版/（v1 冻结）、
server/archive、死代码（distill/train/mcts_ov/ov_bridge_client 等 9 个）、
陈旧测试与 training/archive_scripts。需要考古时在本地找，不在仓库。
