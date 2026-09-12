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
from az_model import (build_net, Head, load_weights_bin, save_weights_bin,
                      C_IN, PCH, N_POS, N_ATTN_EXTRA, FrozenBN)

# 数据路径：默认全部走持久盘（Temp 曾多次被系统清理）；可用环境变量覆盖
BASE = 'D:/data/新建文件夹/chess_game'
TEACHER = os.environ.get('CHESS10_TEACHER', BASE + '/training/teacher')
AZOV = os.environ.get('CHESS10_AZOV', BASE + '/training/data')
SERVER = BASE + '/server'
W_IN = os.environ.get('CHESS10_W_IN', os.path.join(SERVER, 'weights_ov.bin'))
W_OUT = os.environ.get('CHESS10_W_OUT', os.path.join(SERVER, 'weights_ov_new.bin'))

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

# ---------- v3 消融臂开关（最终写进权重 flags，运行时按 flags 走对应图）----------
ARM = {
    'plevel': os.environ.get('CHESS10_ARM_PLEVEL', '0') == '1',   # MANO 按级仿射参与训练
    'window': os.environ.get('CHESS10_ARM_WINDOW', '1') != '0',   # 0 = L0 也做全图注意力
    'attn': int(os.environ.get('CHESS10_ARM_ATTN', '0')),         # 额外末段注意力层数 0/1/2
    'norm': int(os.environ.get('CHESS10_ARM_NORM', '0')),         # 1 = 真 BN 统计（running stats）
}
net.n_attn_extra = max(0, min(N_ATTN_EXTRA, ARM['attn']))
net.mano.window_on = ARM['window']
net.norm_mode = ARM['norm']
net.mano_plevel_trainable = ARM['plevel']
for _m in net.modules():
    if isinstance(_m, FrozenBN):
        _m.norm_mode = ARM['norm']
if not ARM['plevel']:
    net.mano.plg.requires_grad_(False); net.mano.plb.requires_grad_(False)
# 额外层启用但内部权重全零（旧文件零填充）→ 重新小随机初始化，避免 Wo=0 造成的死分支
for _i in range(net.n_attn_extra):
    if float(net.attnX[_i].Wq.weight.abs().sum()) == 0.0:
        for _mm in (net.attnX[_i].Wq, net.attnX[_i].Wk, net.attnX[_i].Wv, net.attnX[_i].Wff1):
            torch.nn.init.normal_(_mm.weight, std=0.05)
        print(f'ARM: attnX[{_i}] 内部权重全零 → 重新小随机初始化（避免死分支）', flush=True)
print(f'ARM: plevel={ARM["plevel"]} window={ARM["window"]} '
      f'attn_extra={net.n_attn_extra} norm={ARM["norm"]}', flush=True)

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

# ---------- AZ 自对弈数据（第二条线，GPU 产出；默认关）----------
# 与 FSF 线分开目录与命名（sp*_encs.f32），避免轮次号撞车；总量由 CHESS10_AZ_CAP 上限控制
AZ_DIR = os.environ.get('CHESS10_AZ_DIR', '')
AZ_CAP = int(os.environ.get('CHESS10_AZ_CAP', '0'))
az_used = 0
if AZ_DIR and AZ_CAP > 0 and os.path.isdir(AZ_DIR):
    az_files = sorted(glob.glob(os.path.join(AZ_DIR, 'sp*_encs.f32')),
                      key=lambda q: os.path.getmtime(q), reverse=True)
    for ef in az_files:
        n = load_f32(ef).size // (C_IN * 100)
        if az_used + n > AZ_CAP and az_used > 0: break
        stem = ef[:-len('_encs.f32')]
        Xe.append(load_f32(ef).reshape(-1, C_IN, 10, 10))
        Xp.append(load_f32(stem + '_pis.f32').reshape(-1, PCH * N_POS))
        Xz.append(load_f32(stem + '_zs.f32'))
        az_used += n

encs = np.concatenate(Xe); pis = np.concatenate(Xp); zs = np.concatenate(Xz)
N = len(zs)
print(f'R{ROUND} data: total={N} (fsf={sp_used} az={az_used} teacher_cap={TEACHER_CAP})', flush=True)

X = torch.from_numpy(encs)
Pi = torch.from_numpy(pis)
# 胜负标签软化：±1 clamp 到 ±0.75（atanh≈±0.973，与教师锚点量级配平，防漂移）
Zt = torch.from_numpy(np.arctanh(np.clip(zs, -Z_CLAMP, Z_CLAMP)).astype(np.float32))

# ---------- 训练（真全参数：主干在计算图内，三组分层 LR；新模块 wd=0） ----------
# v3：额外注意力层与 MANO 按级仿射归入"新模块"（全量 LR、wd=0，与 GRN/MANO 同待遇）；
#     norm=1（真 BN 统计）时 m,v 由 momentum 更新 running stats，不进优化器（也免受 weight_decay 影响）
_frozen_bn_ids = {id(p) for m in net.modules() if isinstance(m, FrozenBN) for p in (m.m, m.v)}
_skip = _frozen_bn_ids if ARM['norm'] == 1 else set()
new_params = [p for n_, p in net.named_parameters() if p.requires_grad and id(p) not in _skip
              and (n_.startswith(('mano.', 'rpb', 'attnX')) or '.grn.' in n_)]
new_ids = {id(p) for p in new_params}
head_params = [p for n_, p in net.named_parameters() if p.requires_grad and id(p) not in _skip
               and not n_.startswith(('conv0', 'bn0', 'blocks')) and id(p) not in new_ids]
trunk_params = [p for n_, p in net.named_parameters() if p.requires_grad and id(p) not in _skip
                and n_.startswith(('conv0', 'bn0', 'blocks')) and id(p) not in new_ids]
_allt = {id(p) for p in net.parameters() if p.requires_grad}
_ungrouped = _allt - {id(p) for g in (new_params, head_params, trunk_params) for p in g}
print(f'PARAMS: head={sum(p.numel() for p in head_params)} '
      f'trunk={sum(p.numel() for p in trunk_params)} new={sum(p.numel() for p in new_params)} '
      f'未入组={len(_ungrouped)}（norm=1 时为 m,v running stats，属预期）', flush=True)
opt = torch.optim.AdamW([
    {'params': head_params, 'lr': LR},
    {'params': trunk_params, 'lr': LR * 0.2},   # 主干小步慢走，头快速适配
    {'params': new_params, 'lr': LR, 'weight_decay': 0.0},   # 新模块（MANO/rpb/GRN）全量 LR，无衰减
], weight_decay=1e-4)
BS = 256 if ON_GPU else 128   # GPU 大批量（用户游戏服务共用 8GB 显存，256 保守安全）
steps_per_ep = (N + BS - 1) // BS
total_steps = steps_per_ep * EPOCHS
WARMUP_STEPS = min(20, max(5, total_steps // 20))
import math
def lr_scale(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    # warmup 后 cosine 衰减至 10% 基频
    prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
t0 = time.time()
trained_eps = 0
gstep = 0
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
        torch.nn.utils.clip_grad_norm_(head_params + trunk_params + new_params, 0.5)
        sched.step(); gstep += 1
        opt.step()
        totL += loss.item(); totV += lv.item(); totP += lp.item(); nb += 1
    print(f'R{ROUND} epoch {ep+1}: loss={totL/nb:.4f} (value={totV/nb:.4f} policy={totP/nb:.4f}) lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s', flush=True)

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