# chess10 — 10×10 国际象棋（chess10d）训练项目

> 状态速览更新于 2026-09-12 15:25。深度文档见文末索引。

## 当前血统

| 代号 | 来源 | 架构 | 终态 |
|---|---|---|---|
| **BJ-1（北京一号）** | AutoDL 云端（北京A区，2080Ti） | v3（3,300,245 floats，ARCH_FLAGS 入权重） | r208 转正，共 168 快照 |
| **R160** | 本地 5070 训练 | v2（3,033,545 floats） | r160 终态，已停机 |

**对战认证（2026-09-12 凌晨）**：BJ-1(r077) vs R160，6 局 250 sims/步、前 8 步开局随机化——**BJ-1 2 胜 4 和 0 负**，两胜均执黑将杀（24/78 步）。工具：`server/match_bj1_vs_r160.js` + `server/match_worker.js`（双 worker 各持权重独立 GPU 上下文；⚠️ NEWOK 应答必须消费否则队列死锁；⚠️ 温度 0 全确定会把同色局下成逐字重演，必须开局随机化）。

## 云端训练线（AutoDL）

- 实例：北京A区 357 机，12 核 8255C + RTX 2080 Ti，包年包月。
- **2026-09-12 15:26 到期未续费 → 关机**；数据保留 15 天（约至 09-27），续费开机即可恢复。
- 循环：`loop.sh`（FSF 教师蒸馏 120 局/轮 + 4 epoch 训练 + 门禁转正）+ `az_loop.sh`（GPU 自对弈混入，AZ_CAP=2000/轮），cwd `/root/autodl-tmp/chess`。
- **恢复命令**（续费开机后）：
  ```bash
  cd /root/autodl-tmp/chess && nohup bash loop.sh >> logs/loop_boot.log 2>&1 &
  cd /root/autodl-tmp/chess && nohup bash az_loop.sh >> logs/az_boot.log 2>&1 &
  ```
- ISA：官方 v14 LB 构建 + `vnni512`（+1.0%，Nodes searched 全等验证后上线；官方 bmi2 备份在位）。

## ⚠️ 训练健康：门禁连续拦截（关机前最新状态）

白优探针一路下行 1.313 → 0.601 → **0.108（r210，已跌破 0.15 门禁线）**；r209、r210 连续两轮未转正（此前 r206 也失败）。门禁行为正确（作废轮次、保留旧权重），但意味着 value 头在向"和棋/低分"方向漂移，疑似与 AZ 混入数据 z=0（和棋标签）占比及截断局启发式标签有关。**续费恢复训练前建议先处理**：复核探针口径 / 调 AZ_CAP / 排查 value 标签分布，否则循环会持续空转烧卡。

## 本地产物（`cloud_pull/`）

- `server/weights_ov.bin`：最新转正权重（sha256 逐次校验）；15:00 定时取回已落地 e59e3db0…（≈r208）。
- `snapshots/`：r001–r208 中全部成功轮快照（含关机前抢拉的 r204/205/207/208）。
- `BJ1/`：取回进程结束后自动打包 `chess10d_v3_BJ1_rNNN.bin` + 血统 README。
- 取回入口：`cloud/pull_results.sh`（幂等；计划任务 `chess10_pull_0312` 已于 15:00 触发完毕，`C:\Users\glowlake\chess10_waiter_1500.sh` 为其等待进程）。

## 2026-09-11~12 已验证的关键修复

1. **教师脚本 750ms 固定睡眠 → 事件驱动**：bestmove 实测 153ms 到达，改 stdout 唤醒 + 超时兜底 + stop 后清缓冲。教师 24.8 分 → 5 分 37 秒（**4.4×**），整轮 ~6.5 分钟。
2. **mcts.js GPU 上传鸡生蛋死锁**：`loadWeights` 依赖 `isReady()`（需 weightsGPU）而 selfplay_gen 只 init 不 upload → 静默 CPU 回退。`gpu.js` 增 `hasContext()` 修复，自对弈实测 GPU 47–50%。
3. **路径可移植化**：FSF/权重路径支持 `CHESS10_*` 环境变量 + 平台感知（Linux 云端可跑）。
4. **`pull_results.sh` 权重漏更 bug**：v3 权重每轮内容变但尺寸恒 13,200,980 字节，"按大小跳过"永远漏更 → 改每次必重下 + sha256 门禁。

## 已知问题

- `单文件版/`（chess10_server.js + chess10.html）为 v1 编码，被 `housekeeping.js` 的 `V2_FROZEN=true` **故意冻结**，不可与新权重混用；主前端 `chess10.html` 为 Magic V2 + NPU 选项（`mcts-npu` 双池路由），不受影响。
- 本地守护 09-02 停机于 r160（终态）；恢复 = 删 `training/STOP.flag` + `training/start_daemon.ps1`。
- v3 遗留测试夹具问题（与生产无关）见 `server/tools/v3_contract.md` §6。

## 文档索引

- v3 架构契约与验收证据：`server/tools/v3_contract.md`
- 模型规格演进（v1→v2→v3）：见 ZCode 记忆 `chess10-model-architecture`
- 云端部署/取回/ISA 实测全记录：见 ZCode 记忆 `chess10-cloud-training-plan`
