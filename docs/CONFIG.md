# chess10d 配置参考（CONFIG）

> 全部配置走环境变量，无配置文件。分组列出：变量 → 默认值 → 含义（读取方）。
> 路径类统一约定：`CHESS10_ROOT` 覆盖仓库根（默认从代码位置自动推导）。

## 路径与根

| 变量 | 默认 | 含义 |
|---|---|---|
| `CHESS10_ROOT` | 自动（`server/paths.js` / `az_model._repo_root()`） | 仓库根覆盖；非标准布局必设 |

## 对弈服务（server/server.js、worker.js）

| 变量 | 默认 | 含义 |
|---|---|---|
| `PORT` | 8787 | HTTP/WS 端口 |
| `CHESS10_WEIGHTS` | `server/weights_ov.bin` → `weights.bin` | 生产权重文件 |
| `CHESS10_WORKERS` | 4 | GPU 池 worker 数 |
| `CHESS10_NPU_WORKERS` | 1 | NPU 池 worker 数（强制 openvino 后端） |
| `CHESS10_ROLE` | - | server.js 自分叉为 worker 时用 |
| `CHESS10_BACKEND` | gpu（opencl） | worker 推理后端：`openvino` 切 NPU |
| `CHESS10_DEVICE` | auto | OpenCL 设备名过滤（`CPU` 特殊值） |
| `CHESS10_FSF` | `fsf/fairy-stockfish(.exe)` | FSF 可执行文件（服务器侧陪玩引擎） |
| `CHESS10_GPU` / `CHESS10_CNN` / `CHESS10_WCNN` / `CHESS10_BATCH` | - | worker 旧版微调钮 |

## OpenVINO 桥（server/ov_bridge_server_v2.py + openvino.js）

| 变量 | 默认 | 含义 |
|---|---|---|
| `CHESS10_OV_BATCH` | 64 | 吞吐档批大小 |
| `CHESS10_OV_BATCH_LAT` | 8 | 延迟档批大小 |
| `CHESS10_OV_DEVICE` | NPU | 编译设备（`GPU.0` = Arc iGPU，`CPU` 兜底） |
| `CHESS10_OV_TURBO` | off | NPU 拉频（实测无收益，留档） |
| `CHESS10_OV_NOFOLD` / `CHESS10_OV_NONATIVE` | off | 关闭 BN 折叠 / 原生批触发（调试用） |
| `CHESS10_OV_INT8` | off | INT8 管线（实测负优化，留档） |
| `OV_BRIDGE_DEBUG` | off | 桥逐段耗时日志 |

## 训练（ov_train/torch_ov_train.py）

| 变量 | 默认 | 含义 |
|---|---|---|
| `CHESS10_W_IN` / `CHESS10_W_OUT` | `server/weights_ov.bin` / 同目录 new | 输入/输出权重 |
| `CHESS10_TEACHER` | `training/teacher` | 教师蒸馏数据目录 |
| `CHESS10_AZOV` | `training/data` | FSF 自对弈数据目录 |
| `CHESS10_AZ_DIR` / `CHESS10_AZ_CAP` | - / 0 | 混入 GPU 自对弈数据（目录/局面数上限） |
| `CHESS10_MANO_CUDA` | off | 启用手写 CUDA 融合 MANO 算子（需 nvcc+MSVC） |
| `CHESS10_ECORE_MASK` | - | 训练绑核 |
| `CHESS10_ARM_{PLEVEL,WINDOW,ATTN,NORM}` | 0 | v3 架构臂开关（消融用） |
| `CHESS10_FORCE_CPU` | off | 剖析脚本强制 CPU |

## 数据生成（server/selfplay_*.js、export_for_pytorch.js）

| 变量 | 默认 | 含义 |
|---|---|---|
| `CHESS10_TEACHER` | 同上 | 导出教师数据目录 |
| `CHESS10_PROBE_W` | - | 探针权重 |
| `CHESS10_AZ_BATCH` / `CHESS10_AZ_FLUSH` | 64 / 200 | 自对弈 MCTS 批参数 |
| `CHESS10_FSF` / `CHESS10_FSF_CORES` | 同上 / 全核 | FSF 路径 / 并行核数 |

## 守护循环（training/*.ps1，PowerShell 纯 ASCII 源）

| 变量 | 默认 | 含义 |
|---|---|---|
| `CHESS10_FSFGAMES` / `CHESS10_FSFEPOCHS` / `CHESS10_FSFMT` / `CHESS10_FSFHOURS` | 120 / 4 / 150 / 4.5 | 每轮局数 / epoch / movetime / 时长 |
| `CHESS10_KEEP_ROUNDS` | 6 | 数据轮换保留数 |

## 对战工具（server/match_*.js）

| 变量 | 默认 | 含义 |
|---|---|---|
| `CHESS10_W_BJ1` | cloud_pull 最新 → `weights/BJ1_r208_v3.bin` | match_bj1_vs_r160 的 BJ-1 方权重 |
