# -*- coding: utf-8 -*-
# NPU b1 档门禁失败诊断：b1 vs b8 vs fp32 参照，同 4 探针局面
# 判据：b1 特征 Δ ≈ b8 特征 Δ → fp16 精度近并列翻转；b1 Δ >> b8 → b1 图结构问题
# 用法: py -3 ov_b1_diag.py [权重]
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, '..'))   # tests/ → 仓库根
sys.path.insert(0, os.path.join(BASE, 'ov_train'))
import az_model as A
from az_model import C_IN, C_HID, N_POS, HEADS, HD, D_MODEL
import openvino as ov
import torch.nn as nn

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'cloud_pull', 'small_extract', 'server', 'weights_ov.bin')
CACHE = os.path.join(BASE, 'server', 'models_ov', 'b1diag')
os.makedirs(CACHE, exist_ok=True)
os.environ.setdefault('OMP_NUM_THREADS', '8')

net = A.build_net(W).eval()
probe = json.load(open(os.path.join(BASE, 'training', 'teacher', 'parity.json')))
px = np.array(probe['encs'], dtype=np.float32)[:4].reshape(-1, C_IN, 10, 10)

# ---- 与桥一致的 BN 折叠 ----
def fold_bn_weights(n):
    s = n.bn0.g.view(-1) / torch.sqrt(n.bn0.v.view(-1) + 1e-5)
    n.conv0.weight.data *= s.view(-1, 1, 1, 1)
    n.conv0.bias.data = n.conv0.bias.data * s + (n.bn0.b.view(-1) - n.bn0.m.view(-1) * s)
    for blk in n.blocks:
        for conv, bn in ((blk.conv1, blk.bn1), (blk.conv2, blk.bn2)):
            s2 = bn.g.view(-1) / torch.sqrt(bn.v.view(-1) + 1e-5)
            conv.weight.data *= s2.view(-1, 1, 1, 1)
            conv.bias.data = conv.bias.data * s2 + (bn.b.view(-1) - bn.m.view(-1) * s2)
    for mod in (n.bn0,) + tuple(b.bn1 for b in n.blocks) + tuple(b.bn2 for b in n.blocks):
        mod.g.data.fill_(1); mod.b.data.fill_(0); mod.m.data.fill_(0); mod.v.data.fill_(1)

fold_bn_weights(net)

class TrunkOnly(nn.Module):
    def __init__(s_, n): super().__init__(); s_.net = n
    def forward(s_, x): return s_.net.trunk(x)

class NativeWrap(nn.Module):
    def __init__(s_, base): super().__init__(); s_.base = base
    def forward(s_, x):
        out = s_.base(x)
        return out, out.mean(dim=0)      # 哑输出（无 batch 轴）

class TorchHead(nn.Module):
    """桥同款：尾段 fp32（transformer + 双头）"""
    def __init__(s_, n):
        super().__init__(); s_.net = n
    def forward(s_, f):
        n = s_.net
        t = f.flatten(2).transpose(1, 2)
        q, k, v = n.Wq(t), n.Wk(t), n.Wv(t)
        q = q.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        k = k.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        v = v.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (HD ** 0.5)
        scores = scores + n.rpb[:, n.idx_r, n.idx_c].unsqueeze(0)
        att = torch.softmax(scores, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(-1, N_POS, D_MODEL)
        h = n.ln1(n.Wo(o) + t)
        h2 = n.ln2(n.ff2(F.relu(n.ff1(h))) + h)
        for i in range(getattr(n, 'n_attn_extra', 0)):
            h2 = n.attnX[i](h2)
        feat = h2.transpose(1, 2).reshape(-1, C_HID, 10, 10)
        pol = n.polo(F.relu(n.polc(feat))).flatten(1)
        hv = F.relu(n.valc(feat)).flatten(1)
        raw = n.vall2(F.relu(n.vall1(hv))).squeeze(-1)
        return pol, raw

head = TorchHead(net).eval()

with torch.no_grad():
    x4 = torch.from_numpy(px)
    feats_ref = net.trunk(x4).numpy()
    pol_ref, raw_ref = head(torch.from_numpy(feats_ref))
    pol_ref = pol_ref.numpy()

print(f'权重: {os.path.basename(W)}  v3={getattr(net, "n_attn_extra", 0) >= 0 and "flags感知"}')
print(f'探针: {px.shape}')

core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(CACHE, 'cache')})

def latency_props(bs):
    p = {'PERFORMANCE_HINT': 'LATENCY'}
    if bs == 1:
        p['NPU_COMPILATION_MODE_PARAMS'] = 'performance-hint-override=latency'
        p['ENABLE_CPU_PINNING'] = 'YES'
    return p

def run_bs(bs):
    m = ov.convert_model(NativeWrap(TrunkOnly(net)).eval(),
                         example_input=torch.zeros(bs, C_IN, 10, 10),
                         input=(bs, C_IN, 10, 10))
    t0 = time.time()
    c = core.compile_model(m, 'NPU', latency_props(bs))
    ct = time.time() - t0
    q = c.create_infer_request()
    outs = [o for o in c.outputs]
    feat_out = outs[0]
    feats = []
    for i in range(4):                      # 逐局面：b1 无 pad；b>1 pad 到 bs 后取前 1
        xb = px[i:i + 1]
        if bs > 1:
            xb = np.concatenate([xb] + [np.zeros((1, C_IN, 10, 10), np.float32)] * (bs - 1))
        it = q.get_input_tensor(); it.data[:] = xb
        q.infer()
        feats.append(np.array(q.get_tensor(feat_out).data, copy=True)[0])
    return np.stack(feats), ct

print('\n=== 特征层对比（trunk 输出，fp32 参照）===')
results = {}
for bs in (1, 8):
    f, ct = run_bs(bs)
    d = np.abs(f - feats_ref)
    results[bs] = f
    print(f'b{bs}: 编译 {ct:.1f}s  特征 max|Δ|={d.max():.5f}  mean|Δ|={d.mean():.6f}')

print('\n=== 策略层对比（TorchHead fp32 尾段）===')
for bs in (1, 8):
    with torch.no_grad():
        pol, raw = head(torch.from_numpy(results[bs]))
    pol = pol.numpy()
    am_new = pol.argmax(1); am_ref = pol_ref.argmax(1)
    # 参照的 top1/top2 边际（softmax 前的 logits 差）
    for i in range(4):
        srt = np.sort(pol_ref[i])[::-1]
        margin = srt[0] - srt[1]
        flag = 'OK ' if am_new[i] == am_ref[i] else 'FLIP'
        print(f'b{bs} pos{i}: argmax ref={am_ref[i]:5d} npu={am_new[i]:5d} {flag}  ref_margin={margin:.4f}  '
              f'logitΔ@ref_argmax={abs(pol[i, am_ref[i]] - pol_ref[i, am_ref[i]]):.4f}')

print('\n判读：若 b1 特征Δ ≈ b8 特征Δ 且翻转处 ref_margin 极小（<0.05）→ fp16 近并列，属精度噪声；'
      '若 b1 特征Δ 显著大于 b8 → b1 图结构问题。')
