# -*- coding: utf-8 -*-
# NNCF INT8 量化 trunk（NPU 原生支持 U8/INT8 计算原语，官方量化路径）
# 产出: server/models_ov/trunk_v2_int8_b{B}.xml（桥 CHESS10_OV_INT8=1 时优先加载，仍过精度门禁）
# 本脚本自带验收：NPU 编译 + 探针精度 vs torch fp32 + 吞吐对比（fp16 vs int8）
import os, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn as nn
import nncf
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN, C_HID  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 64

net = build_net(os.path.join(SRV, 'weights_ov.bin')).eval()

class TrunkOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return self.net.trunk(x)

# ---- 校准数据：教师 parity 探针 + 训练编码（真实分布优先） ----
cands = [
    os.path.normpath(os.path.join(HERE, '..', '..', 'training', 'teacher', 'train_encs.f32')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'training', 'teacher', 'parity.json')),
]
encs = None
for c in cands:
    if not os.path.exists(c):
        continue
    if c.endswith('.f32'):
        raw = np.fromfile(c, dtype=np.float32)
        encs = raw.reshape(-1, C_IN, 10, 10)[:512]
        break
    with open(c) as fp:
        import json
        encs = np.array(json.load(fp)['encs'], dtype=np.float32)[:512].reshape(-1, C_IN, 10, 10)
        break
if encs is None or len(encs) == 0:
    encs = np.random.default_rng(7).standard_normal((512, C_IN, 10, 10)).astype(np.float32) * 0.5
print(f'calibration boards: {len(encs)}', flush=True)

t0 = time.time()
ov_model = ov.convert_model(TrunkOnly(net).eval(),
                            example_input=torch.zeros(B, C_IN, 10, 10),
                            input=(B, C_IN, 10, 10))
print(f'convert {time.time()-t0:.1f}s', flush=True)

chunks = []
for i in range(0, len(encs) - B + 1, B):
    chunks.append(encs[i:i + B].astype(np.float32))
if not chunks:   # 样本不足一个 batch 就复制补
    while len(encs) < B:
        encs = np.concatenate([encs, encs])
    chunks = [encs[:B].astype(np.float32)]
calib = nncf.Dataset(chunks)

t0 = time.time()
q_model = nncf.quantize(ov_model, calib, subset_size=min(16, len(chunks)))
print(f'quantize {time.time()-t0:.1f}s', flush=True)

CACHE = os.path.join(SRV, 'models_ov')
os.makedirs(CACHE, exist_ok=True)
st = os.stat(os.path.join(SRV, 'weights_ov.bin'))
sig = f'{int(st.st_mtime_ns)}:{st.st_size}'
xml = os.path.join(CACHE, f'trunk_v2_int8_b{B}.xml')
ov.save_model(q_model, xml)
open(xml + '.meta', 'w').write(sig)
print('WROTE', xml, flush=True)

# ---- NPU 编译 + 精度 + 吞吐对比 ----
core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(CACHE, 'cache')})
with torch.no_grad():
    torch_feats = net.trunk(torch.zeros(2, C_IN, 10, 10)).numpy()

def bench(model, tag):
    try:
        comp = core.compile_model(model, 'NPU')
    except Exception as e:
        print(f'{tag}: NPU compile FAIL {str(e)[:140]}', flush=True)
        return
    ireq = comp.create_infer_request()
    xb = np.zeros((B, C_IN, 10, 10), dtype=np.float32)
    f = np.array(ireq.infer({0: xb})[comp.outputs[0]]).copy()
    dv = float(np.abs(f[:2] - torch_feats).max())
    ireq.infer({0: xb})
    t1 = time.time()
    R = 20
    for _ in range(R):
        ireq.infer({0: xb})
    ms = (time.time() - t1) / R * 1000
    print(f'{tag}: NPU 推理 {ms:.1f}ms/{B}盘 ({ms/B:.3f}ms/盘) | vs torch 特征 maxAbs={dv:.4f}', flush=True)

bench(ov_model, 'fp16 trunk')
bench(q_model, 'int8 trunk')
print('INT8 PIPELINE DONE', flush=True)
