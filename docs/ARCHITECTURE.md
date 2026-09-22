# chess10d 架构说明（ARCHITECTURE）

> 2026-09-23 架构现代化定稿。本文回答三件事：系统怎么分层、多实现怎么不漂移、数据怎么流。

## 1. 分层总览

```
┌─ 对弈层 ───────────────────────────────────────────────┐
│ web/chess10.html（单文件前端）   desktop/winui（WinUI 壳）│
└──────────────┬────────────────────────────────────────┘
               │ WebSocket
┌─ 服务层 ─────▼─────────────────────────────────────────┐
│ server/server.js  静态托管 + WS 路由 + 双池（GPU 池/NPU 池）│
│ server/worker.js  引擎工作进程（fork，按池分组）           │
└──────┬─────────────────────────────────────────────────┘
       │
┌─ 搜索与规则 ──▼─────────────────────────────────────────┐
│ server/mcts.js    MCTS（TT 跨步树复用、Dirichlet、温度采样）│
│                  └ moveChannel/encodeBoardInt＝编码唯一实现 │
│ server/engine.js  10×10 规则（纯逻辑零依赖，被 20+ 模块引用）│
└──────┬─────────────────────────────────────────────────┘
       │ evalBatch(boards[N][107])
┌─ 推理后端（三选一，接口同形）───────────────────────────────┐
│ server/gpu.js      OpenCL GPU（kernel 正文在 server/cl/）  │
│ server/openvino.js OpenVINO NPU/iGPU（py 桥 ov_bridge_*.py）│
│ server/cnn.js      JS CPU 前向（零依赖兜底；权重格式权威定义）│
└──────┬─────────────────────────────────────────────────┘
       │ 权重文件 = 唯一事实源（float32 数组 + ARCH_FLAGS 尾部）
┌─ 训练层 ──────▼─────────────────────────────────────────┐
│ ov_train/az_model.py        PyTorch 权威实现（训练用）      │
│ ov_train/torch_ov_train.py  CUDA 训练器（门禁转正制）       │
│ ov_train/mano_cuda.py       手写 CUDA 融合 MANO 算子(可选)  │
│ server/selfplay_fsf_teacher.js  FSF 教师蒸馏数据线          │
│ server/selfplay_gen.js          MCTS 自对弈数据线           │
└──────┬─────────────────────────────────────────────────┘
┌─ 运维层 ──────▼─────────────────────────────────────────┐
│ training/*.ps1 守护循环   cloud/*.sh 云端时代脚本（历史）    │
└─────────────────────────────────────────────────────────┘
```

路径约定：全仓唯一允许解析仓库根的代码是 `server/paths.js`（JS）与
`az_model._repo_root()`（Python）；`CHESS10_ROOT` 环境变量可覆盖一切。
新增代码禁止硬编码绝对路径。

## 2. 多实现一致性：本项目最核心的工程决策

同一个网络存在 **4 份前向实现**（PyTorch / OpenCL / OpenVINO-桥 / JS CPU）。
防漂移的机制不是"少写实现"，而是三层锁：

1. **权重即接口**：裸 float32 数组、按长度自动识别 v1/v2/v3、尾部 16 floats
   ARCH_FLAGS 携带架构开关（policy 通道数/按级仿射/窗口开关/额外注意力层数/归一化模式）。
   所有实现从同一文件读同一布局，格式权威定义在 `server/cnn.js` 与 `ov_train/az_model.py`。
2. **两两对拍门禁**（tests/run_all.js）：

   | 对拍 | 门禁 | 历史基线 |
   |---|---|---|
   | JS ↔ PyTorch（v3 全宽） | max\|Δ\| | 1.3e-6 |
   | GPU ↔ JS CPU | 相对 Δ ≤1e-5 量级 | 7.0e-6 |
   | OpenVINO(NPU fp16) ↔ JS | value≤0.05 / policy≤3.0 / margin-aware top1 | b32 96ms |
   | 手写 CUDA kernel ↔ eager | fwd≤1e-3 / grad≤1e-2 | 实测 1e-6 级 |
   | **集成级**：patch 后整模块 ↔ eager | Δ≤1e-3 | 见下 |

3. **契约文档**：docs/contracts/v2_contract.md、v3_contract.md——布局、flags、
   验收证据、已知问题的唯一书面权威。

**集成级门禁的由来**（2026-09-16 教训）：mano_cuda 的 kernel 级对拍全过
（Δ≈1e-6），但 patch 集成分支的反窗 permute 布局错误把输出打乱——kernel 对拍
覆盖不到集成分支。自此后凡 monkeypatch/替换类改动必过
`tests/integration_gate.py`（patch 前后整模块输出对拍）。

## 3. 门禁式推进（训练安全模型）

```
教师/自对弈数据 → torch_ov_train（4 epoch）→ 探针三题（初始/白优/中局）
  → verify_weights（前向有限性）→ 双通过 → 转正 weights_ov.bin + 快照
  → 任一不过 → 本轮作废，生产权重不动
```

- 探针语义见 docs/RUNBOOK.md。历史上真实拦截：NORM 臂（loss 10→82）、
  r209/r210 白优探针漂移。
- margin-aware top1（2026-09-17 修正）：参照自身 top1-top2 logits 差 <0.05 的
  位置属并列，argmax 掷硬币不构成失败——v3 策略头概率更平后此修正是必要的。

## 4. 关键设计决策与理由

| 决策 | 理由 | 代价（已知债务） |
|---|---|---|
| 权重=裸 float32 数组 | 四种语言生态零依赖直读；v1/v2/v3 按长度识别 | 无 schema，改布局必须同步多实现（靠契约+对拍兜住） |
| 4 份前向实现 | 训练(CUDA)/对弈(OpenCL)/省电(NPU)/兜底(JS)各取所长 | 每处架构改动×4；靠对拍矩阵压漂移 |
| OpenCL kernel 外置 .cl | kernel 可独立阅读编辑；#define 绑定层留 JS 与 cnn 常量联动 | 多一个运行时文件依赖 |
| monkeypatch 式可选加速（CHESS10_MANO_CUDA=1） | 默认路径零风险，opt-in 吃优化 | patch 正确性需集成门禁专门守护 |
| 双池 worker（GPU 池+NPU 池） | 对弈时路由 mcts/mcts-npu，NPU 省电陪玩不占 GPU | 进程管理复杂度 |
| 守护用 ps1（Windows）/sh（云端） | 平台原生、免额外依赖 | 两套运维脚本并存 |

## 5. 性能决策档案（全部实测，否决项留档）

- AMP fp16 + fused AdamW：训练步 62→43ms（−30%），**唯一大头收益**
- 手写 CUDA 融合 MANO：kernel 切片 2.12×，整网 fp32 −4.6%；attention 仅占
  整步 ~1%，AMP 路径与 eager 持平 → 已知天花板
- SDPA（官方融合核）竞品线：fp16 切片 0.308ms 最快，整网持平，留作对照
- 已实测否决：NPU INT8 量化（更慢）、NPU TURBO/AsyncInferQueue、PGO（−2.6%）、
  GPU 侧 BN 折叠（无收益）、FSF AVX-512 SWAR 四合一、torch.compile（Windows 无 Triton）

## 6. 已知债务清单（诚实版）

1. 4 份前向实现的维护乘数（架构改动的固有成本）
2. JS 侧反向传播不含 attnX/plg（v3 契约 §6；训练在 PyTorch 侧，无实际影响）
3. gpu_v2_parity 步骤 2/4 历史失败（测试夹具与权重不配对，WARN 处理，见 v3 契约）
4. cloud/*.sh 面向已到期的 AutoDL 实例（含硬编码云端路径），作为历史运维档案保留
5. chess10.html 2427 行单文件（部署特性：单文件可玩，刻意不为拆分而拆分）
