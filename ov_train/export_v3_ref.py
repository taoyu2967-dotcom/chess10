# -*- coding: utf-8 -*-
# v3 跨语言对拍参考生成器（Python 权威 = az_model.py）
#
# 产出到 server/tools/v3_ref/：
#   weights_v3_rand.bin      —— v3 尾部（Wp2x/plg/plb/attnX）全部非平凡随机；flags=[3,160,1,1,2,0]
#   weights_v3_v2equiv.bin   —— v2 等价配置：新张量恒等（Wp2x 语义化 warm-init、plg=1/plb=0、attnX 零）；flags=[3,160,0,1,0,0]
#   weights_v3_ablate.bin    —— 消融：flags=[3,160,0,0,2,0]（按级仿射冻结但 plg/plb 非平凡；MANO 窗口关；额外层 2 层）
#   encs.f32                 —— 固定输入 32×24×10×10（通道主序，与 encodeBoard 一致）
#   pol_logits.f32 / value_raw.f32                     （rand，全宽 16000）
#   pol_logits_v2equiv.f32 / value_raw_v2equiv.f32     （v2 等价）
#   pol_logits_ablate.f32 / value_raw_ablate.f32       （消融）
#   pol_logits_r160.f32 / value_raw_r160.f32           （真实 r160.bin 经 v3 warm-init）
#   manifest.json
#
# 数值口径：CPU、torch.set_num_threads(1)、torch.no_grad()（与 JS forwardCPU 对拍用）
# 用法: py -3 export_v3_ref.py
import os
import sys
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CHESS10_ROOT') or os.path.dirname(HERE)   # 路径中枢约定
sys.path.insert(0, HERE)
import az_model as A

torch.set_num_threads(1)
torch.set_grad_enabled(False)

OUT = os.path.join(BASE, 'server', 'tools', 'v3_ref')
os.makedirs(OUT, exist_ok=True)
BASE_W = os.path.join(BASE, 'server', 'weights_ov.bin')          # v2 生产权重（新模块零/恒等）
R160 = os.path.join(BASE, 'training', 'data', 'snapshots', 'r160.bin')
N = 32


def load_inputs():
    f = os.path.join(BASE, 'training', 'teacher', 'train_encs.f32')
    encs = np.fromfile(f, dtype=np.float32).reshape(-1, A.C_IN, 10, 10)[:N]
    assert encs.shape[0] == N, encs.shape
    return encs


def run_and_dump(net, encs, tag):
    X = torch.from_numpy(encs.reshape(-1, A.C_IN, 10, 10))
    with torch.no_grad():
        pol, raw = net(X)
    pol = pol.detach().cpu().numpy().astype(np.float32)
    raw = raw.detach().cpu().numpy().astype(np.float32)
    assert pol.shape == (N, A.PCH * A.N_POS), pol.shape
    pol.tofile(os.path.join(OUT, f'pol_logits{tag}.f32'))
    raw.tofile(os.path.join(OUT, f'value_raw{tag}.f32'))
    return pol, raw


def main():
    encs = load_inputs()
    encs.tofile(os.path.join(OUT, 'encs.f32'))
    meta = {'n': int(N), 'C_IN': A.C_IN, 'PCH': A.PCH, 'N_POS': A.N_POS,
            'POL': int(A.PCH * A.N_POS), 'V3_FLOATS': int(A.V3_FLOATS),
            'files': ['encs.f32'], 'cases': {}}

    # ---------------- 1) 非平凡 v3：随机化 policy 新行 + 按级仿射 + 额外层 ----------------
    raw = A.load_weights_bin(BASE_W)
    assert raw['__v2__'] and not raw['__v3__']
    raw['__v3__'] = True
    raw['flags'] = raw['flags'].copy()
    raw['flags'][A.FLAG_IDX['plevel']] = 1.0   # 按级仿射可训练
    raw['flags'][A.FLAG_IDX['window']] = 1.0   # MANO 窗口开
    raw['flags'][A.FLAG_IDX['attn_extra']] = 2.0
    raw['flags'][A.FLAG_IDX['norm']] = 0.0
    net = A.Net(raw).eval()

    g = torch.Generator().manual_seed(20260911)
    with torch.no_grad():
        # policy 新 60 行
        net.polo.weight[A.PCH_LEGACY:].normal_(0.0, 0.05, generator=g)
        net.polo.bias[A.PCH_LEGACY:].normal_(0.0, 0.02, generator=g)
        # MANO 按级仿射（非平凡）
        net.mano.plg.normal_(1.0, 0.1, generator=g)
        net.mano.plb.normal_(0.0, 0.1, generator=g)
        # 额外末段注意力层：所有参数非零（含 LayerNorm）
        for L in net.attnX:
            for m in (L.Wq, L.Wk, L.Wv, L.Wo, L.Wff1, L.Wff2):
                m.weight.normal_(0.0, 0.05, generator=g)
            L.Wff1.bias.normal_(0.0, 0.02, generator=g)
            L.Wff2.bias.normal_(0.0, 0.02, generator=g)
            L.ln1.weight.normal_(1.0, 0.05, generator=g); L.ln1.bias.normal_(0.0, 0.05, generator=g)
            L.ln2.weight.normal_(1.0, 0.05, generator=g); L.ln2.bias.normal_(0.0, 0.05, generator=g)

    p = os.path.join(OUT, 'weights_v3_rand.bin')
    n = A.save_weights_bin(net, p)
    assert n == A.V3_FLOATS, (n, A.V3_FLOATS)
    print(f'[1] weights_v3_rand.bin: {n} floats == V3_FLOATS {A.V3_FLOATS}', flush=True)
    # 从落盘文件重载后前向（保证 bin ↔ f32 参考严格对应）
    net_r = A.build_net(p).eval()
    ts = np.fromfile(p, dtype=np.float32)
    fl = ts[A.V3_FLOATS - A.FLAG_N:].copy()
    print(f'    flags = {np.array2string(fl[:6], precision=1)}', flush=True)
    run_and_dump(net_r, encs, '')
    meta['cases']['rand'] = {'weights': 'weights_v3_rand.bin',
                             'flags': [float(v) for v in fl[:6]],
                             'pol': 'pol_logits.f32', 'raw': 'value_raw.f32'}
    meta['files'] += ['pol_logits.f32', 'value_raw.f32']

    # ---------------- 2) v2 等价配置：flags=[3,160,0,1,0,0] ----------------
    raw2 = A.load_weights_bin(BASE_W)
    raw2['__v3__'] = True
    raw2['flags'] = A.default_flags()
    raw2['flags'][A.FLAG_IDX['plevel']] = 0.0
    raw2['flags'][A.FLAG_IDX['window']] = 1.0
    raw2['flags'][A.FLAG_IDX['attn_extra']] = 0.0
    raw2['flags'][A.FLAG_IDX['norm']] = 0.0
    net2 = A.Net(raw2).eval()
    p2 = os.path.join(OUT, 'weights_v3_v2equiv.bin')
    n2 = A.save_weights_bin(net2, p2)
    assert n2 == A.V3_FLOATS, (n2, A.V3_FLOATS)
    fl2 = np.fromfile(p2, dtype=np.float32)[A.V3_FLOATS - A.FLAG_N:]
    print(f'[2] weights_v3_v2equiv.bin: {n2} floats  flags = {np.array2string(fl2[:6], precision=1)}', flush=True)
    net2_r = A.build_net(p2).eval()
    run_and_dump(net2_r, encs, '_v2equiv')
    meta['cases']['v2equiv'] = {'weights': 'weights_v3_v2equiv.bin',
                                'flags': [float(v) for v in fl2[:6]],
                                'pol': 'pol_logits_v2equiv.f32', 'raw': 'value_raw_v2equiv.f32'}
    meta['files'] += ['pol_logits_v2equiv.f32', 'value_raw_v2equiv.f32']

    # ---------------- 3) 消融：flags[2]=0（按级仿射"冻结"但值非平凡）且 flags[3]=0（MANO 窗口关）------------
    # 目的：证明按级仿射"无论 flags[2] 如何都照常应用"，以及窗口开关真的切换到单窗=全图。
    # 注意：az_model.save_weights_bin 的 net_flags() 把 window 固定写 1，故落盘后手工把 flags[3] 改成 0。
    raw4 = A.load_weights_bin(BASE_W)
    raw4['__v3__'] = True
    raw4['flags'] = A.default_flags()
    raw4['flags'][A.FLAG_IDX['plevel']] = 0.0   # 训练侧冻结（但 plg/plb 仍非恒等）
    raw4['flags'][A.FLAG_IDX['window']] = 0.0   # MANO L0 也走单窗=全图
    raw4['flags'][A.FLAG_IDX['attn_extra']] = 2.0
    raw4['flags'][A.FLAG_IDX['norm']] = 0.0
    net4 = A.Net(raw4).eval()
    g4 = torch.Generator().manual_seed(424242)
    with torch.no_grad():
        net4.polo.weight[A.PCH_LEGACY:].normal_(0.0, 0.05, generator=g4)
        net4.polo.bias[A.PCH_LEGACY:].normal_(0.0, 0.02, generator=g4)
        net4.mano.plg.normal_(1.0, 0.15, generator=g4)   # 非平凡
        net4.mano.plb.normal_(0.0, 0.15, generator=g4)
        for L in net4.attnX:
            for m in (L.Wq, L.Wk, L.Wv, L.Wo, L.Wff1, L.Wff2):
                m.weight.normal_(0.0, 0.05, generator=g4)
            L.Wff1.bias.normal_(0.0, 0.02, generator=g4)
            L.Wff2.bias.normal_(0.0, 0.02, generator=g4)
            L.ln1.weight.normal_(1.0, 0.05, generator=g4); L.ln1.bias.normal_(0.0, 0.05, generator=g4)
            L.ln2.weight.normal_(1.0, 0.05, generator=g4); L.ln2.bias.normal_(0.0, 0.05, generator=g4)
    p4 = os.path.join(OUT, 'weights_v3_ablate.bin')
    n4 = A.save_weights_bin(net4, p4)
    assert n4 == A.V3_FLOATS, (n4, A.V3_FLOATS)
    buf = np.fromfile(p4, dtype=np.float32)
    buf[A.V3_FLOATS - A.FLAG_N + A.FLAG_IDX['window']] = 0.0   # 手工关窗
    buf.tofile(p4)
    fl4 = np.fromfile(p4, dtype=np.float32)[A.V3_FLOATS - A.FLAG_N:]
    print(f'[3] weights_v3_ablate.bin: {n4} floats  flags = {np.array2string(fl4[:6], precision=1)}', flush=True)
    net4_r = A.build_net(p4).eval()
    assert net4_r.mano.window_on is False, 'window flag patch failed'
    run_and_dump(net4_r, encs, '_ablate')
    meta['cases']['ablate'] = {'weights': 'weights_v3_ablate.bin',
                               'flags': [float(v) for v in fl4[:6]],
                               'pol': 'pol_logits_ablate.f32', 'raw': 'value_raw_ablate.f32'}
    meta['files'] += ['pol_logits_ablate.f32', 'value_raw_ablate.f32']

    # ---------------- 4) 真实权重 r160（v2 文件）→ v3 warm-init 全宽参考 ----------------
    raw3 = A.load_weights_bin(R160)
    assert raw3['__v2__'] and not raw3['__v3__']
    fl3 = raw3['flags']
    print(f'[4] r160.bin: v2 文件 → warm-init flags = {np.array2string(fl3[:6], precision=1)}', flush=True)
    net3 = A.Net(raw3).eval()
    run_and_dump(net3, encs, '_r160')
    # warm-init 尾段参考（用于 JS 侧逐位断言：Wp2x/bp2x/plg/plb）
    raw3['Wp2x'].astype(np.float32).tofile(os.path.join(OUT, 'wp2x_r160.f32'))
    raw3['bp2x'].astype(np.float32).tofile(os.path.join(OUT, 'bp2x_r160.f32'))
    raw3['plg'].astype(np.float32).tofile(os.path.join(OUT, 'plg_r160.f32'))
    raw3['plb'].astype(np.float32).tofile(os.path.join(OUT, 'plb_r160.f32'))
    meta['cases']['r160'] = {'weights': '../training/data/snapshots/r160.bin',
                             'flags': [float(v) for v in fl3[:6]],
                             'pol': 'pol_logits_r160.f32', 'raw': 'value_raw_r160.f32',
                             'warm_tail': ['wp2x_r160.f32', 'bp2x_r160.f32', 'plg_r160.f32', 'plb_r160.f32']}
    meta['files'] += ['pol_logits_r160.f32', 'value_raw_r160.f32',
                      'wp2x_r160.f32', 'bp2x_r160.f32', 'plg_r160.f32', 'plb_r160.f32']

    with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    print('ref written ->', OUT, flush=True)


if __name__ == '__main__':
    main()
