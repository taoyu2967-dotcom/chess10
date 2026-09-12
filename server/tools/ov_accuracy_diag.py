# -*- coding: utf-8 -*-
# 精度二分诊断：torch fp32 基准 vs 同一份 IR 的 CPU 插件 vs NPU
# 若 IR@CPU ≈ torch 而 IR@NPU 差 → fp16 精度问题；若 IR@CPU 也差 → 导出语义问题
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN, PCH, N_POS  # noqa: E402

# ---- 盘源：教师 parity 探针（真实局面）+ 受控随机 ----
import json
with open(os.path.normpath(os.path.join(HERE, '..', '..', 'training', 'teacher', 'parity.json'))) as fp:
    encs = np.array(json.load(fp)['encs'], dtype=np.float32)
x = torch.from_numpy(encs[:4].reshape(-1, C_IN, 10, 10))

net = build_net(os.path.join(HERE, '..', 'weights_ov.bin')).eval()
with torch.no_grad():
    pol_t, raw_t = net(x)
pol_t = pol_t.numpy(); val_t = np.tanh(raw_t.numpy())

B = encs[:4].shape[0]
print('converting IR (b32)...', flush=True)
m = ov.convert_model(net.eval(), example_input=torch.zeros(B, C_IN, 10, 10),
                     input=(B, C_IN, 10, 10))
print(f'IR converted, outputs={len(m.outputs)}', flush=True)

def run(device):
    compiled = ov.Core().compile_model(m, device)
    res = compiled({0: x.numpy().astype(np.float32)})
    pol_o = val_o = None
    for k, v in res.items():
        sh = v.shape
        if len(sh) == 2 and sh[1] == PCH * N_POS: pol_o = v   # v3 策略宽度 = 160*100 = 16000
        elif len(sh) == 1: val_o = v
    return np.array(val_o).reshape(-1), np.array(pol_o)

for dev in ('CPU', 'NPU'):
    try:
        val_o, pol_o = run(dev)
        dv = np.abs(val_o - val_t).max()
        dp = np.abs(pol_o - pol_t).max()
        top1 = float((pol_o.argmax(1) == pol_t.argmax(1)).mean())
        print(f'{dev}: value maxAbs={dv:.4f}  policy maxAbs={dp:.4f}  top1={top1:.3f}', flush=True)
        print(f'     torch vals={np.round(val_t,3)}  {dev} vals={np.round(val_o,3)}', flush=True)
    except Exception as e:
        print(f'{dev}: FAIL {str(e)[:200]}', flush=True)
print('DIAG DONE', flush=True)
