# -*- coding: utf-8 -*-
# 导出跨语言对拍参考：随机化新模块的 v2 权重 + 固定输入的前向激活（.f32 + manifest）
# 供 cnn.js / gpu.js 移植验证（布局错误、实现错误都会在这里现形）
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'D:/data/新建文件夹/chess_game'
sys.path.insert(0, HERE)
import az_model

OUT = os.path.join(BASE, 'server', 'tools', 'v2_ref')
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 随机化新模块的 v2 权重（seed 固定；JS 侧无需复现随机，直接用导出的 bin）
torch.manual_seed(20260901)
net = az_model.build_net(os.path.join(BASE, 'server', 'weights_ov.bin')).to(dev)
with torch.no_grad():
    net.mano.Wq.weight.normal_(0, 0.02); net.mano.Wk.weight.normal_(0, 0.02)
    net.mano.Wv.weight.normal_(0, 0.02); net.mano.Wo.weight.normal_(0, 0.02)
    net.mano.down.weight.normal_(0, 0.05); net.mano.up.weight.normal_(0, 0.05)
    net.mano.ln.weight.normal_(1, 0.02); net.mano.ln.bias.normal_(0, 0.02)
    net.rpb.normal_(0, 0.05)
    for blk in net.blocks:
        blk.grn.gamma.normal_(0, 0.05)
wpath = os.path.join(OUT, 'weights_v2_rand.bin')
n = az_model.save_weights_bin(net, wpath)
print(f'v2 rand weights: {n} floats -> {wpath}', flush=True)

# 2) 参考激活：32 个真实局面编码（前 16 教师随机 + 16 个手工局面含易位/ep/升变态）
encs = np.fromfile(os.path.join(BASE, 'training', 'teacher', 'train_encs.f32'), dtype=np.float32).reshape(-1, 24, 10, 10)[:32]
X = torch.from_numpy(encs).to(dev)
with torch.no_grad():
    feat = net.trunk(X)                                   # (32,128,10,10) MANO+GRN 后
    pol, raw = net(X)
feat.cpu().numpy().tofile(os.path.join(OUT, 'feat_trunk.f32'))
pol.cpu().numpy().tofile(os.path.join(OUT, 'pol_logits.f32'))
raw.cpu().numpy().tofile(os.path.join(OUT, 'raw_value.f32'))
encs.tofile(os.path.join(OUT, 'inputs.f32'))

# 3) 中间参考：MANO 输出（喂相同 trunk 前3块输出）
with torch.no_grad():
    f0 = F = net.bn0(net.conv0(X)) if False else None
# 简化：直接重跑前向分段
with torch.no_grad():
    import torch.nn.functional as Fp
    f = Fp.relu(net.bn0(net.conv0(X)))
    for blk in net.blocks[:3]:
        f = blk(f)
    f_after_grn1 = f.clone()
    f_mano = net.mano(f)
    f_mano.cpu().numpy().tofile(os.path.join(OUT, 'mano_out.f32'))

manifest = {
    'n': int(len(encs)), 'C_IN': 24, 'C_HID': 128, 'HEADS': 4, 'D_MODEL': 128,
    'PCH': 100, 'N_POS': 100, 'MANO_WINDOW': 5, 'MANO_LEVELS': 3,
    'files': ['inputs.f32', 'feat_trunk.f32', 'pol_logits.f32', 'raw_value.f32', 'mano_out.f32'],
    'weightFile': 'weights_v2_rand.bin',
    'note': 'feat_trunk = trunk 全部输出(3CNN+MANO+3CNN+GRN)；mano_out = MANO 模块输出（输入=前3块后特征）',
}
with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as fp:
    json.dump(manifest, fp, ensure_ascii=False, indent=1)
print('ref activations written:', OUT, flush=True)

# 4) rpb 隔离参考：只有 rpb 随机化（MANO/GRN 全零恒等）→ pol 差异只来自 rpb 布局
net2 = az_model.build_net(os.path.join(BASE, 'server', 'weights_ov.bin')).to(dev)
torch.manual_seed(777)
with torch.no_grad():
    net2.rpb.normal_(0, 0.05)
w2 = os.path.join(OUT, 'weights_v2_rpbonly.bin')
az_model.save_weights_bin(net2, w2)
with torch.no_grad():
    pol2, raw2 = net2(X)
pol2.cpu().numpy().tofile(os.path.join(OUT, 'pol_logits_rpbonly.f32'))
print('rpb-only reference written', flush=True)
