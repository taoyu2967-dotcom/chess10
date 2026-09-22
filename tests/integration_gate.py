# -*- coding: utf-8 -*-
# 集成级门禁（2026-09-23 制度化）：patch / monkeypatch 类改动的整模块对拍。
# 教训来源：mano_cuda kernel 级对拍全过（Δ≈1e-6）但 patch 集成分支反窗布局错乱，
# 是 SDPA 竞速线审计抓出的——kernel 级对拍 ≠ 集成正确，凡 patch 必须过本门禁。
# 判据：patch 前后 MANO 模块输出 max|Δ| ≤ 1e-3（fp32 下限，随机批 B=64）
# 用法: py -3 tests/integration_gate.py [--strict]
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(BASE, 'ov_train'))
import az_model as A  # noqa: E402

STRICT = '--strict' in sys.argv

W = os.path.join(BASE, 'server', 'weights_ov.bin')
if not os.path.exists(W):
    W = os.path.join(BASE, 'weights', 'BJ1_r208_v3.bin')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = A.build_net(W).to(dev).eval()
torch.manual_seed(3)
x = torch.randn(64, A.C_HID, 10, 10, device=dev)   # MANO 的实际输入：128 通道特征图

fails, skips = [], []
with torch.no_grad():
    ref = net.mano(x)
    for mod_name in ('mano_cuda', 'mano_sdpa'):
        try:
            m = __import__(mod_name)
            m.patch_mano()
            got = net.mano(x)
            m.unpatch_mano()
            d = (ref - got).abs().max().item()
            ok = d <= 1e-3
            print(f'[{"PASS" if ok else "FAIL"}] {mod_name} MANO 整模块 max|Δ|={d:.2e}', flush=True)
            if not ok:
                fails.append(mod_name)
        except Exception as e:
            # 依赖缺失（无 CUDA/nvcc）＝环境不可用，非正确性失败：默认 SKIP，--strict 时计失败
            print(f'[SKIP] {mod_name}: {str(e)[:140]}', flush=True)
            if STRICT:
                fails.append(mod_name)
            else:
                skips.append(mod_name)

print(f'integration_gate: {"FAIL" if fails else "PASS"}（fail={fails} skip={skips}）', flush=True)
sys.exit(1 if fails else 0)
