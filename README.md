# chess10d — 10×10 国际象棋变体训练与对弈套件

> AlphaZero 风格自研训练管线 + GPU/NPU/CPU 三后端推理 + 网页对弈界面

![License](https://img.shields.io/badge/license-GPL--3.0-blue) ![Node](https://img.shields.io/badge/node-%E2%89%A518-green) ![GPU](https://img.shields.io/badge/GPU-OpenCL%E5%8F%AF%E9%80%89-orange)

**chess10d** 是在 Fairy-Stockfish `variants.ini` 中定义的 10×10 国际象棋变体（含自定义棋子 `d` 与扩展规则，走法详见 [`fsf/variants.ini`](fsf/variants.ini)）。本仓库包含围绕它构建的完整套件：

- **训练管线**：Fairy-Stockfish 教师蒸馏 + GPU 自对弈（MCTS）→ PyTorch CUDA 训练 → 门禁核验 → 权重转正
- **两个推理后端**：OpenCL GPU（默认）/ OpenVINO（NPU，双池路由）+ 纯 JS CPU 兜底
- **对弈界面**：单文件网页前端 [`chess10.html`](chess10.html)，接入服务器 MCTS 引擎或浏览器本地 stockfish.js；另附 WinUI 3 桌面壳
- **对战评测工具**：任意两套权重自动化对战（执先轮换、开局随机化、比分统计）

## 预训练模型（Pretrained Models）

权重为自定义二进制格式（float32 数组），v1/v2/v3 三代布局均可在运行时自动识别加载（含尾部 ARCH_FLAGS 校验）。

| 文件 | 架构 | 参数量 (floats) | 训练来源 | 说明 |
|---|---|---|---|---|
| [`weights/BJ1_r208_v3.bin`](weights/BJ1_r208_v3.bin) | v3 | 3,300,245 | 云端 RTX 2080 Ti（r001–r208 蒸馏+自对弈混训） | **当前最强**。r077 版本对 R160 六局 2 胜 4 和 0 负（两胜执黑将杀） |
| [`weights/R160_v2.bin`](weights/R160_v2.bin) | v2 | 3,033,545 | 本地 RTX 5070（r140–r160） | 本地训练线 v2 架构终态 |
| [`weights/v3_arm0_local.bin`](weights/v3_arm0_local.bin) | v3 | 3,300,245 | 本地 RTX 5070（v3 迁移基线） | 仅含 policy 编码修复的对照基线（arm0） |
| [`weights/v1_legacy_local.bin`](weights/v1_legacy_local.bin) | v1 | 2,834,213 | 本地 GPU 基线时代 | 早期 v1 架构遗留权重 |

> 架构演进：v2 引入 MANO 窗口注意力/GRN/rpb 主干；v3 修复 policy 编码非单射缺陷（POLICY_CH 100→160：基础走法 0–99、马连跳 100–131、升变 132–146、兜底 147），并把架构开关（ARCH_FLAGS）写进权重尾部，保证 JS / PyTorch / GPU 三实现不漂移。三实现对拍：JS↔PyTorch max|Δ|=1.3e-6，GPU↔CPU 7.6e-6。详见 [`server/tools/v3_contract.md`](server/tools/v3_contract.md)。

## 快速开始（对弈）

```bash
# 1. 准备权重：从 weights/ 选一个复制为服务端生产权重
cp weights/BJ1_r208_v3.bin server/weights_ov.bin

# 2. 启动服务
node server/server.js

# 3. 浏览器打开
#    http://127.0.0.1:8787/chess10.html
```

## 推理后端

| 后端 | 设备 | 启用方式 | 说明 |
|---|---|---|---|
| **OpenCL GPU**（默认） | NVIDIA / AMD / Intel GPU | 开箱即用 | fp32，MCTS 批推理（`opencl-raub`） |
| **OpenVINO** | Intel NPU / Arc iGPU / CPU | `CHESS10_BACKEND=openvino` | fp16 + 精度门禁（value≤0.05 / policy≤3.0 / top1≥99%，不过门禁自动降级）；`CHESS10_NPU_WORKERS` 控制 NPU 池大小，与 GPU 池双池并行路由 |
| **JS CPU**（兜底） | 任意 CPU | 无 GPU 时自动 | 纯 JS 前向，保证零依赖可跑 |

前端界面内可切换：**MCTS（OpenCL GPU 池）** / **MCTS-NPU（OpenVINO 池）** / **本地引擎**（浏览器内 stockfish.js，无需服务端）。

## 桌面壳（WinUI 3）

`winui/` 提供一个 WinUI 3 + WebView2 的桌面封装：启动时自动拉起 `server/server.js`（Node），窗口内加载对弈前端，关闭时自动回收服务进程。若仓库里找不到 `server/server.js`，壳会退化为纯浏览器模式，连接已在外部启动的服务。

```bash
dotnet build winui/Chess10d.csproj -c Release -p:Platform=x64
winui\bin\x64\Release\net8.0-windows10.0.19041.0\Chess10d.exe
```

依赖 .NET 8 SDK 构建、Windows 10 19041+ 运行（WebView2 运行时 Windows 11 自带）。PRI 打包走 `Microsoft.Windows.SDK.BuildTools` 包内工具链（`EnableMsixTooling`），无需 Visual Studio。

## 对战评测

```bash
node server/match_bj1_vs_r160.js 6 250
```

双进程各持一套权重（独立 GPU 上下文），250 sims/步，前 8 步温度 1.0 开局随机化，300 步截断判和。改文件头部 `W_BJ1` / `W_R160` 两个路径即可换成任意两套权重对战。

## 训练

训练管线分三条数据线，每轮数据汇总后由 [`ov_train/torch_ov_train.py`](ov_train/torch_ov_train.py) 训练，经探针门禁 + `verify_weights.js` 双核验才转正：

| 环节 | 入口 | 说明 |
|---|---|---|
| 教师蒸馏 | `server/selfplay_fsf_teacher.js` | Fairy-Stockfish v14（大棋盘构建）MultiPV 走子做策略蒸馏，事件驱动 UCI（bestmove 到达即续） |
| 自对弈 | `server/selfplay_gen.js` | MCTS 自对弈（OpenCL GPU 批推理），产 sp*_encs/pis/zs 三元组 |
| 训练 | `ov_train/torch_ov_train.py` | CUDA，环境变量 `CHESS10_W_IN/W_OUT` 控制输出，`CHESS10_AZ_DIR/AZ_CAP` 混入自对弈数据 |
| 云端一键循环 | `cloud/loop.sh` + `cloud/az_loop.sh` | Linux 无人值守：每轮教师 120 局 → 训练 4 epoch → 门禁 → 转正+快照 |

**依赖**：Node ≥18；可选 CUDA（PyTorch）与 OpenCL（`opencl-raub`）；教师引擎需自备 [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) v14 大棋盘构建（变体定义已含于 `fsf/variants.ini`，二进制不入库）。

## 仓库结构

```
chess10.html            # 网页对弈前端（单文件，含 Magic V2 UI）
server/                 # 引擎、MCTS、GPU(OpenCL)/NPU(OpenVINO) 后端、蒸馏与自对弈
  tools/                # 对拍测试、对战工具、架构契约文档 (v2/v3_contract.md)
ov_train/               # PyTorch 训练器与架构定义 (az_model.py)
training/               # 本地守护进程、数据治理、损失面可视化
cloud/                  # 云端无人值守训练循环与产物取回脚本
fsf/variants.ini        # chess10d 变体定义（Fairy-Stockfish）
weights/                # 预训练模型（见上表）
lib/                    # 前端依赖库（浏览器本地引擎 stockfish.js、chess.min.js）
winui/                  # WinUI 3 + WebView2 桌面壳（自动拉起/回收 node 服务端）
单文件版/                # 早期 v1 单文件打包（遗留，不与新权重兼容）
```

## License

[GPL-3.0](LICENSE)。`lib/stockfish.js` 为 GPL-3.0 组件（Stockfish 的 WASM 构建），随仓库一并分发。
