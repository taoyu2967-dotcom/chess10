# 10×10 象棋 · GPU 神经网络引擎（AlphaZero 架构）

## 游戏规则（10×10 扩展版）

- 棋盘 10×10，列 a-j，行 1-10（黑方在 1 行侧，白方在 10 行侧）。初始布局 `drnbqkbnrd` / 白 `DRNBQKBNRD`。
- 棋子：`p`兵 `n`马 `b`象 `r`车 `q`后 `k`王 `d`炮兵（Dabbaba）。
- **炮兵 d**：前后左右直跳 2 或 3 格，可越子，落点吃子。
- **马 n**：日字跳（可越子）；**连跳特权（仅限起始格）**：马位于本方起始格（白 c10/h10，黑 c1/h1）时，本回合可一步连跳两个日字（纯飞跃：不检查路径/拐马脚、不要求中间落点为空，仅要求最终落点非己方子且不吃王）；离开起始格后按普通马处理。
- **象 b**：斜走任意格；**走日特权**：对方皇后不在场上时，可额外走马步（日字），可越子、落点吃子。
- **吃过路兵**：兵从起始行连走两格后形成目标格（FEN ep 字段）。小兵仅限对方连走两格后的立即一步可吃（FEN 带 `!` 后缀标记）；**炮兵特权**：只要受害兵仍在场上，任意位置的炮兵可随时跳到目标格将其吃掉（无视 2/3 格距离限制）。
- **易位**：王 f→h(王侧)/f→d(后侧)，车 i→g/b→e。标准规则：王动过或车动过/车在起始格被吃则清除对应权；王所经/所到格不受攻击。
- 兵升变：到达底线可升变为 后/车/象/马/炮兵。
- 和棋：逼和、三次重复、无子可胜（单轻子）、50 步规则（半回合 ≥100）。
- 王不能被吃：走法生成与攻击检测均不产生吃王着法，王被将死即负。

## 网络架构（残差CNN + 注意力 + 双头）

```
输入棋盘特征 (B, 24, 10, 10)
  ├─ 通道: 0-13 己方/敌方棋子(7+7), 14 行棋方, 15 己王受威胁,
  │        16 EP目标, 17-20 易位权(KQkq), 21-23 预留
  ↓
Conv3×3(24→128) → BN → ReLU
  ↓
6× ResBlock(Conv3×3→BN→ReLU → Conv3×3→BN → 残差+ReLU)
  ↓
reshape (B, 128, 10, 10) → (B, 100, 128)  token 序列
  ↓
单层 4 头全局自注意力(d=128, ff=256) + 残差 + LayerNorm ×2
  ↓
还原 (B, 100, 128) → (B, 128, 10, 10)
  ├─ Policy头: Conv3×3(128→32)→BN→ReLU → Conv1×1(32→100方向平面logits)
  └─ Value头:  Conv3×3(128→32)→BN→ReLU → Linear(3200→256)→ReLU → Linear(256→1)→tanh
```

- 策略平面 100 通道：72 滑行方向-距离(8方向×9) + 8 马跳 + 8 炮兵(4方向×2距离) + 4 兵(前2/斜吃×2/EP)
- 训练损失：Policy 交叉熵 + 0.25×Value MSE（AlphaZero 标准组合）
- MCTS-PUCT：网络 Policy 为先验 P，Value 为叶子评估，cPuct=2.5，广度优先铺宽

## 文件

| 文件 | 说明 |
|---|---|
| `cnn.js` | 网络定义 + CPU 前向（forwardCPU/trunkForward/batchHeads）+ 权重存取 |
| `gpu.js` | OpenCL 全流水线批量推理（FP32，每 kernel 同步） |
| `mcts.js` | MCTS-PUCT + 方向平面策略先验 + 预测池 |
| `engine.js` | 本地引擎（规则 + 启发式评估） |
| `train.js` | AlphaZero 自对弈训练（GPU 自对弈 → CPU 双头训练） |
| `distill.js` | 教师蒸馏训练（本地引擎评估当标签，学得更快） |
| `worker.js` | 搜索子进程（独立 GPU context） |
| `server.js` | WS 服务（engine: mcts / fsf） |
| `test.js` | GPU/CPU 一致性 + MCTS 回归测试 |

## 运行

```bash
npm install        # 安装 opencl-raub、ws
node test.js       # 自测（GPU vs CPU 一致性 + MCTS）
node server.js     # 启动服务（8787 端口）
node train.js 8 8 120   # AlphaZero 自对弈训练（epochs 局数 每步模拟）
node distill.js 12 400 20  # 教师蒸馏训练（epochs 局面数 最大步数）
```

环境变量：`CHESS10_DEVICE`（GPU 选择）、`CHESS10_WORKERS`（默认4）、`CHESS10_WCNN`（CNN权重占比，默认：有 weights.bin 时 0.7，否则 0.25）、`CHESS10_BATCH`（默认512）。

## 训练

1. 首次运行无权重 → 随机初始化，MCTS 以启发式为主（wCnn=0.25）
2. `node distill.js 12 400 20` 快速蒸馏本地引擎知识（几分钟）
3. `node train.js 8 8 120` 自对弈强化（时间较长）
4. 训练权重存 `weights.bin`，重启 server 自动加载，wCnn 自动调高到 0.7

训练提示：自对弈数据目前多为长和棋（胜负信号弱），建议先用 distill 打底再自对弈强化；价值/策略头解析梯度训练（主干冻结），主干训练需扩展反向传播。
