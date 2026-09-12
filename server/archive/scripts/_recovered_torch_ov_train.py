# -*- coding: utf-8 -*-
# OV 版 CPU 训练器 v3：全参数训练（主干解冻）+ 胜负标签软化 + 训后探针门禁
# 修复清单（对应日志诊断）：
#   P0-2  坏权重死循环 → 训后门禁不达标直接作废本轮（保留旧权重），不再依赖下一轮 FATAL
#   P1-3  policy 不动/主干冻结 → 主干解冻全参数训练，policy 权重 2.0
#   P1-5  value 单调向负漂移 → 胜负标签 clamp ±0.75（atanh≈±0.97，与教师锚点量级配平），
#         LR 2e-4→1e-4，梯度裁剪 1.0→0.5，value 权重 1.3
#   P1-6  白优坍缩 → 教师数据继续混合锚定 + 训后白优 raw 须 > 0.15
# 用法: py torch_ov_train.py [epochs] [lr] [round]
import glob, json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F

# 设备：GPU 优先（FSF 教师数据训练无自对弈循环，无需省电，直接吃 RTX）
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ON_GPU = dev.type == 'cuda'

# E 核亲和：仅 CPU 训练时启用（夜间自对弈循环省电用）；GPU 训练不钉核，保证数据供给
E_MASK = 0 if ON_GPU else int(os.environ.get('CHESS10_ECORE_MASK', '1020'))
if E_MASK and sys.platform == 'win32':
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = k32.GetCurrentProcess()
        if k32.SetProcessAffinityMask(handle, E_MASK):
            torch.set_num_threads(min(8, bin(E_MASK).count('1')))
            print(f'AFFINITY: E核掩码 {E_MASK} (CPU 2-9), torch threads={torch.get_num_threads()}', flush=True)
        else:
            print(f'AFFINITY: SetProcessAffinityMask 失败 err={ctypes.GetLastError()}', flush=True)
    except Exception as _e:
        print(f'AFFINITY: 异常 {_e}', flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from az_model import build_net, Head, load_weights_bin, save_weights_bin, C_IN, PCH, N_POS

TEACHER = 'C:/Users/glowlake/AppData/Local/Temp/opencode/pytorch_data'
AZOV = 'C:/Users/glowlake/AppData/Local/Temp/opencode/az_data_ov'
SERVER = 'D:/data/新建文件夹/chess_game/server'
W_IN = os.path.join(SERVER, 'weights_ov.bin')
W_OUT = os.path.join(SERVER, 'weights_ov_new.bin')

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
ROUND = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TEACHER_CAP, SP_CAP, TIME_CAP = 6000, 8000, 420
V_W, P_W = 1.3, 2.0          # value/policy 损失权重（policy 加权推动学习）
Z_CLAMP = 0.75               # 胜负标签软化：atanh(0.75)≈0.973，与教师 raw 量级配平，防漂移

net = build_net(W_IN).to(dev)
if ON_GPU:
    torch.backends.cudnn.benchmark = True
    print(f'DEVICE: {torch.cuda.get_device_name(0)}', flush=True)
head = Head(net)   # 主干解冻：全参数训练

# 哨兵：训练前探针（有限 + 白优为正）
with open(os.path.join(TEACHER, 'parity.json')) as fp:
    pe = np.array(json.load(fp)['encs'], dtype=np.float32)
net.eval()
with torch.no_grad():
    pre = [round(net(torch.from_numpy(e.reshape(1, C_IN, 10, 10)).to(dev))[1].item(), 3) for e in pe[:3]]
print(f'R{ROUND} pre-train probe [初始/白优/中局]: {pre}', flush=True)
if not (np.isfinite(pre).all() and pre[1] > 0.0):
    print(f'R{ROUND} FATAL: 权重加载异常（白优 raw {pre[1]}），中止')
    sys.exit(1)

# ---------- 数据：原始局面编码（主干进训练循环，不再预计算特征） ----------
def load_f32(p): return np.fromfile(p, dtype=np.float32)

Xe, Xp, Xz = [], [], []
te = os.path.join(TEACHER, 'train_encs.f32')
if os.path.exists(te):
    Xe.append(load_f32(te).reshape(-1, C_IN, 10, 10)[:TEACHER_CAP])
    Xp.append(load_f32(os.path.join(TEACHER, 'train_pis.f32')).reshape(-1, PCH * N_POS)[:TEACHER_CAP])
    Xz.append(load_f32(os.path.join(TEACHER, 'train_zs.f32'))[:TEACHER_CAP])
sp_files = sorted(glob.glob(os.path.join(AZOV, 'r*_encs.f32')),
                  key=lambda q: int(re.search(r'r(\d+)_encs', q).group(1)), reverse=True)
sp_used = 0
for ef in sp_files:
    n = load_f32(ef).size // (C_IN * 100)
    if sp_used + n > SP_CAP and sp_used > 0: break
    stem = ef[:-len('_encs.f32')]
    Xe.append(load_f32(ef).reshape(-1, C_IN, 10, 10))
    Xp.append(load_f32(stem + '_pis.f32').reshape(-1, PCH * N_POS))
    Xz.append(load_f32(stem + '_zs.f32'))
    sp_used += n
encs = np.concatenate(Xe); pis = np.concatenate(Xp); zs = np.concatenate(Xz)
N = len(zs)
print(f'R{ROUND} data: total={N} (selfplay={sp_used})', flush=True)

X = torch.from_numpy(encs)
Pi = torch.from_numpy(pis)
# 胜负标签软化：±1 clamp 到 ±0.75（atanh≈±0.973，与教师锚点量级配平，防漂移）
Zt = torch.from_numpy(np.arctanh(np.clip(zs, -Z_CLAMP, Z_CLAMP)).astype(np.float32))

# ---------- 训练（真全参数：主干在计算图内，分层 LR 保护主干特征） ----------
head_params = [p for n_, p in net.named_parameters() if p.requires_grad and not n_.startswith(('conv0', 'bn0', 'blocks'))]
trunk_params = [p for n_, p in net.named_parameters() if p.requires_grad and n_.startswith(('conv0', 'bn0', 'blocks'))]
opt = torch.optim.AdamW([
    {'params': head_params, 'lr': LR},
    {'params': trunk_params, 'lr': LR * 0.2},   # 主干小步慢走，头快速适配
], weight_decay=1e-4)
BS = 256 if ON_GPU else 128   # GPU 大批量（用户游戏服务共用 8GB 显存，256 保守安全）
t0 = time.time()
trained_eps = 0
for ep in range(EPOCHS):
    if time.time() - t0 > TIME_CAP: break
    trained_eps += 1
    net.train()
    perm = torch.randperm(N)
    totL = totV = totP = 0.0; nb = 0
    for i in range(0, N, BS):
        idx = perm[i:i+BS]
        xb = X[idx].to(dev, non_blocking=True)
        pb = Pi[idx].to(dev, non_blocking=True)
        zb = Zt[idx].to(dev, non_blocking=True)
        pol, raw = net(xb)
        lv = F.mse_loss(raw, zb)
        lp = -(pb * F.log_softmax(pol, dim=1)).sum(1).mean()
        loss = V_W * lv + P_W * lp
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params + trunk_params, 0.5)
        opt.step()
        totL += loss.item(); totV += lv.item(); totP += lp.item(); nb += 1
    print(f'R{ROUND} epoch {ep+1}: loss={totL/nb:.4f} (value={totV/nb:.4f} policy={totP/nb:.4f}) {time.time()-t0:.0f}s', flush=True)

# ---------- 训后探针门禁：不达标作废本轮（等效自动回滚，杜绝坏权重死循环） ----------
net.eval()
with torch.no_grad():
    post = [round(net(torch.from_numpy(e.reshape(1, C_IN, 10, 10)).to(dev))[1].item(), 3) for e in pe[:3]]
print(f'R{ROUND} post-train probe [初始/白优/中局]: {post}', flush=True)
ok = (np.isfinite(post).all()
      and post[1] > 0.15                     # 白优仍显著为正（防坍缩）
      and abs(post[0]) < 1.2                 # 初始局面接近均势（防整体偏置）
      and post[1] >= pre[1] - 0.5)           # 单轮漂移上限（真全参数首轮头需适配，适度放宽）
if not ok:
    print(f'R{ROUND} GATE FAIL: 探针不达标（白优 {post[1]}, 初始 {post[0]}），本轮作废保留旧权重', flush=True)
    sys.exit(1)

cnt = save_weights_bin(net, W_OUT)
print(f'R{ROUND} exported weights_ov_new.bin ({cnt} floats, {trained_eps} epochs)', flush=True)