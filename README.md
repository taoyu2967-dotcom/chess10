# chess10d — 10×10 Chess Variant: Training & Play Suite

English | [简体中文](README.zh-CN.md)

> AlphaZero-style self-contained training pipeline + tri-backend inference (GPU/NPU/CPU) + web & desktop play UI

![License](https://img.shields.io/badge/license-GPL--3.0-blue) ![Node](https://img.shields.io/badge/node-%E2%89%A518-green) ![GPU](https://img.shields.io/badge/GPU-OpenCL%E5%8F%AF%E9%80%89-orange)

**chess10d** is a 10×10 chess variant defined in a Fairy-Stockfish [`variants.ini`](fsf/variants.ini) (custom piece `d` and extended rules — see the file for exact movement). This repository contains the complete suite built around it:

- **Training pipeline**: Fairy-Stockfish teacher distillation + GPU self-play (MCTS) → PyTorch CUDA training → acceptance gates → weight promotion
- **Two inference backends**: OpenCL GPU (default) / OpenVINO (NPU, dual-pool routing), plus a pure-JS CPU fallback
- **Play UI**: single-file web frontend [`chess10.html`](chess10.html) driven by the server MCTS engine or in-browser stockfish.js; a WinUI 3 desktop shell is included
- **Match tool**: automated head-to-head between any two weights (color swap, opening randomization, score tracking)

## Pretrained Models

Weights use a custom binary format (float32 arrays). All three layout generations (v1/v2/v3) are auto-detected at load time, with trailing ARCH_FLAGS validation.

| File | Arch | Params (floats) | Trained on | Notes |
|---|---|---|---|---|
| [`weights/BJ1_r208_v3.bin`](weights/BJ1_r208_v3.bin) | v3 | 3,300,245 | Cloud RTX 2080 Ti (r001–r208, distillation + self-play mix) | **Strongest.** Its r077 checkpoint beat R160 2 wins / 4 draws / 0 losses over six games (both wins by checkmate as Black) |
| [`weights/R160_v2.bin`](weights/R160_v2.bin) | v2 | 3,033,545 | Local RTX 5070 (r140–r160) | Final checkpoint of the local v2 line |
| [`weights/v3_arm0_local.bin`](weights/v3_arm0_local.bin) | v3 | 3,300,245 | Local RTX 5070 (v3 migration baseline) | Control baseline containing only the policy-encoding fix (arm0) |
| [`weights/v1_legacy_local.bin`](weights/v1_legacy_local.bin) | v1 | 2,834,213 | Local GPU baseline era | Legacy early v1 weights |

> Architecture evolution: v2 introduced the MANO windowed-attention / GRN / rpb trunk. v3 fixed a policy-encoding non-injectivity bug (POLICY_CH 100→160: base moves 0–99, knight double-jump 100–131, promotions 132–146, fallback 147) and writes architecture switches (ARCH_FLAGS) into the weight tail so the JS / PyTorch / GPU implementations cannot drift. Cross-implementation parity: JS↔PyTorch max|Δ|=1.3e-6, GPU↔CPU 7.6e-6. See [`docs/contracts/v3_contract.md`](docs/contracts/v3_contract.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick Start (Play)

```bash
# 1. Pick a weight file and copy it to the server's production path
cp weights/BJ1_r208_v3.bin server/weights_ov.bin

# 2. Start the server
node server/server.js

# 3. Open in browser
#    http://127.0.0.1:8787/chess10.html
```

## Inference Backends

| Backend | Device | How to enable | Notes |
|---|---|---|---|
| **OpenCL GPU** (default) | NVIDIA / AMD / Intel GPU | Works out of the box | fp32, batched MCTS inference (`opencl-raub`) |
| **OpenVINO** | Intel NPU / Arc iGPU / CPU | `CHESS10_BACKEND=openvino` | fp16 with an accuracy gate (value≤0.05 / policy≤3.0 / top1≥99%, auto-downgrades on failure); `CHESS10_NPU_WORKERS` sizes the NPU pool, routed in parallel with the GPU pool |
| **JS CPU** (fallback) | any CPU | automatic when no GPU | pure-JS forward pass, zero extra dependencies |

The frontend can switch between **MCTS (OpenCL GPU pool)** / **MCTS-NPU (OpenVINO pool)** / **Local engine** (in-browser stockfish.js, no server needed).

## Desktop Shell (WinUI 3)

`winui/` contains a WinUI 3 + WebView2 desktop wrapper: it launches `server/server.js` (Node) automatically, loads the play UI in a native window, and reclaims the server process on close. If `server/server.js` is not found, the shell degrades to pure-browser mode connecting to an externally started server.

```bash
dotnet build winui/Chess10d.csproj -c Release -p:Platform=x64
winui\bin\x64\Release\net8.0-windows10.0.19041.0\Chess10d.exe
```

Requires .NET 8 SDK to build, Windows 10 19041+ to run (WebView2 runtime ships with Windows 11). PRI packaging goes through the toolchain inside the `Microsoft.Windows.SDK.BuildTools` package (`EnableMsixTooling`) — no Visual Studio required.

## Match Tool

```bash
node server/match_bj1_vs_r160.js 6 250
```

Two worker processes each hold one set of weights (independent GPU contexts), 250 sims/move, first 8 moves sampled at temperature 1.0 for opening diversity, 300-move cap adjudicated as a draw. Edit `W_BJ1` / `W_R160` at the top of the file to match any two weights.

## Training

The pipeline runs three data lines. Per-round data is merged and trained by [`ov_train/torch_ov_train.py`](ov_train/torch_ov_train.py); a round is promoted only after passing both the probe gate and `verify_weights.js`:

| Stage | Entry point | Notes |
|---|---|---|
| Teacher distillation | `server/selfplay_fsf_teacher.js` | Policy distillation from Fairy-Stockfish v14 (large-board build) MultiPV moves; event-driven UCI (resumes the moment bestmove arrives) |
| Self-play | `server/selfplay_gen.js` | MCTS self-play (OpenCL GPU batched inference), producing sp*_encs/pis/zs triplets |
| Training | `ov_train/torch_ov_train.py` | CUDA; `CHESS10_W_IN/W_OUT` control outputs, `CHESS10_AZ_DIR/AZ_CAP` mix in self-play data |
| Unattended cloud loop | `cloud/loop.sh` + `cloud/az_loop.sh` | Linux headless: 120 teacher games per round → 4-epoch training → gate → promotion + snapshot |

**Dependencies**: Node ≥18; optional CUDA (PyTorch) and OpenCL (`opencl-raub`); the teacher engine requires a self-provided [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) v14 large-board build (the variant definition is included in `fsf/variants.ini`; binaries are not committed).

## Test & Gates

```bash
npm test            # full gate suite (~5-8 min): encoding injectivity, movegen/attack equivalence,
                    # cnn/GPU parity, e2e smoke (real server + WS), integration gate, OpenVINO×NPU,
                    # weight probes ×4
npm run test:quick  # pure-JS core gates (~1 min, no GPU/python needed)
```

Any change to the network, kernels, or patch layers must keep `npm test` green. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2 for the parity matrix and its thresholds.

## Repository Layout

```
chess10.html            # Web play frontend (single file, Magic V2 UI)
server/                 # Engine, MCTS, backends, distillation & self-play
  cl/kernels.cl         #   OpenCL kernel sources (pure OpenCL C, editable standalone)
  paths.js              #   Path hub — the only place allowed to resolve the repo root
  tools/                #   Bench & diagnostic utilities (gates live in tests/)
tests/                  # Gate suite: run_all.js orchestrates all acceptance gates
ov_train/               # PyTorch trainer & architecture definition (az_model.py)
training/               # Local daemon, data governance, loss-landscape visualization
cloud/                  # Cloud-era ops scripts & artifact pull (historical, see RUNBOOK)
fsf/variants.ini        # chess10d variant definition (Fairy-Stockfish)
weights/                # Pretrained models (table above)
lib/                    # Frontend libraries (in-browser engine stockfish.js, chess.min.js)
winui/                  # WinUI 3 + WebView2 desktop shell (auto-starts/recycles the node server)
docs/                   # ARCHITECTURE / RUNBOOK / CONFIG + contracts (v2/v3)
```

## License

[GPL-3.0](LICENSE). `lib/stockfish.js` is a GPL-3.0 component (WASM build of Stockfish) redistributed with this repository.
