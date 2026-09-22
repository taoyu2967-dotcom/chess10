# -*- coding: utf-8 -*-
# chess10 网络定义 v3（2026-09-11）：v2 + policy 头扩宽(PCH 100→160) + MANO 按级仿射
#   + 末段额外注意力层（零初始化残差式）+ ARCH_FLAGS
#
# 兼容性设计：v1(2,834,213) / v2(3,033,545) 权重仍可加载 ——
#   * 新增张量按"恒等"初始化（按级仿射 γ=1/β=0；额外层 Wo=0、Wff2=0）；
#   * policy 新通道行（100..159）以语义化 warm-init 从最接近的旧通道行复制；
#   * 因此旧权重的**前 100 个通道 logits 逐位不变**（纯增量兼容）。
#
# v3 追加布局（顺序，紧跟 V2_TAIL 之后）：
#   Wp2x(60*32) bp2x(60) | plg(3*128) plb(3*128)
#   | 额外注意力层 ×2，每层：WqX WkX WvX WoX(各128*128) Wff1X(128*256) bff1X(256)
#                             Wff2X(256*128) bff2X(128) ln1gX ln1bX ln2gX ln2bX(各128)
#   | flags(16)
# 权重布局与 server/cnn.js 必须逐项一致。
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _repo_root():
    """路径中枢（Python 侧）：CHESS10_ROOT 优先，否则从本文件向上找仓库锚点（fsf/variants.ini + weights/）。"""
    import os as _os
    if _os.environ.get('CHESS10_ROOT'):
        return _os.path.resolve(_os.environ['CHESS10_ROOT'])
    d = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(6):
        if _os.path.exists(_os.path.join(d, 'fsf', 'variants.ini')) and _os.path.exists(_os.path.join(d, 'weights')):
            return d
        d = _os.path.dirname(d)
    raise RuntimeError('az_model: 未找到仓库根，请设 CHESS10_ROOT')

import os
SERVER = os.path.join(_repo_root(), 'server')
C_IN, C_HID, RES_BLOCKS, HEADS, D_MODEL, D_FF, PCH, N_POS = 24, 128, 6, 4, 128, 256, 160, 100
HD = D_MODEL // HEADS
MANO_WINDOW, MANO_LEVELS = 5, 3
PCH_LEGACY = 100          # v1/v2 的 policy 通道数；新通道行从此之后追加
N_ATTN_EXTRA = 2          # 尾部固定容纳 2 层（实际启用数由 flags 决定，默认 0）
ARCH_VERSION = 3
FLAG_N = 16
V2_TAIL = (6 * C_HID + 4 * D_MODEL * D_MODEL + C_HID * C_HID * 4 + C_HID
           + C_HID * C_HID * 4 + C_HID + D_MODEL + D_MODEL + HEADS * 361)
POLICY_TAIL = (PCH - PCH_LEGACY) * 32 + (PCH - PCH_LEGACY)
PLEVEL_TAIL = MANO_LEVELS * D_MODEL * 2
PER_ATTN = (4 * D_MODEL * D_MODEL + D_MODEL * D_FF + D_FF
            + D_FF * D_MODEL + D_MODEL + 4 * D_MODEL)
ATTN_X_TAIL = N_ATTN_EXTRA * PER_ATTN
V3_TAIL = POLICY_TAIL + PLEVEL_TAIL + ATTN_X_TAIL + FLAG_N
V3_FLOATS = 2834213 + V2_TAIL + V3_TAIL

# 语义化 warm-init：新通道行 ← 最接近的旧通道行
WARM_ROWS = {'djump': 72, 'promo': (0, 89, 90), 'rest': 0}
# flags 位语义：0=版本 1=PCH 2=启用按级仿射(训练侧) 3=MANO 窗口开 4=额外注意力层数 5=归一化模式(训练侧)
FLAG_IDX = {'ver': 0, 'pch': 1, 'plevel': 2, 'window': 3, 'attn_extra': 4, 'norm': 5}


def default_flags():
    f = np.zeros(FLAG_N, np.float32)
    f[FLAG_IDX['ver']] = ARCH_VERSION
    f[FLAG_IDX['pch']] = PCH
    f[FLAG_IDX['plevel']] = 0        # 按级仿射：0=冻结在恒等（Arm 0），1=参与训练
    f[FLAG_IDX['window']] = 1        # MANO L0 开窗（v2 现状）
    f[FLAG_IDX['attn_extra']] = 0    # 额外注意力层数
    f[FLAG_IDX['norm']] = 0          # 0=可学习仿射（旧行为）；1=running stats（真 BN 统计，消融臂）
    return f


def _warm_init_policy(out):
    """v1/v2 权重缺新通道行 → 从旧行复制（连跳←马步 72；升变←兵 0/89/90）。旧通道行不动。"""
    n_new = PCH - PCH_LEGACY
    Wp2 = out['Wp2'].reshape(PCH_LEGACY, 32)
    bp2 = out['bp2']
    Wx = np.zeros((n_new, 32), np.float32)
    bx = np.zeros(n_new, np.float32)
    kn = WARM_ROWS['djump']
    for k in range(32):                       # 通道 100..131：马连跳
        Wx[k] = Wp2[kn]; bx[k] = bp2[kn]
    for kind, row in enumerate(WARM_ROWS['promo']):   # 通道 132..146：升变 3 位移 × 5 子
        for pi in range(5):
            Wx[32 + kind * 5 + pi] = Wp2[row]; bx[32 + kind * 5 + pi] = bp2[row]
    for k in range(32 + 15, n_new):            # 147..159：预留/兜底
        Wx[k] = Wp2[WARM_ROWS['rest']]; bx[k] = bp2[WARM_ROWS['rest']]
    out['Wp2x'] = Wx.flatten(); out['bp2x'] = bx
    out['plg'] = np.ones(MANO_LEVELS * D_MODEL, np.float32)
    out['plb'] = np.zeros(MANO_LEVELS * D_MODEL, np.float32)
    for i in range(N_ATTN_EXTRA):
        # 内部权重用小随机初始化（保证 Wo 有非零梯度），仅输出投影 Wo/Wff2 置零 → 前向恒等
        # 注意：若全部置零会形成"死分支"（Wo=0 → 内层梯度恒为 0，永远学不动）
        rs = np.random.RandomState(1000 + i)
        for k, n in ATT_K_TAIL:
            out[f'attnX{i}_{k}'] = np.zeros(n, np.float32)
        for k, shape, std in [('Wq', (D_MODEL, D_MODEL), 0.05), ('Wk', (D_MODEL, D_MODEL), 0.05),
                              ('Wv', (D_MODEL, D_MODEL), 0.05), ('Wff1', (D_MODEL, D_FF), 0.05)]:
            out[f'attnX{i}_{k}'] = (rs.randn(*shape) * std).astype(np.float32).flatten()
        for k, n in [('ln1g', D_MODEL), ('ln2g', D_MODEL)]:
            out[f'attnX{i}_{k}'] = np.ones(n, np.float32)
    out['flags'] = default_flags()


ATT_K_TAIL = [('Wq', D_MODEL * D_MODEL), ('Wk', D_MODEL * D_MODEL),
              ('Wv', D_MODEL * D_MODEL), ('Wo', D_MODEL * D_MODEL),
              ('Wff1', D_MODEL * D_FF), ('bff1', D_FF),
              ('Wff2', D_FF * D_MODEL), ('bff2', D_MODEL),
              ('ln1g', D_MODEL), ('ln1b', D_MODEL), ('ln2g', D_MODEL), ('ln2b', D_MODEL)]


def load_weights_bin(path):
    flat = np.fromfile(path, dtype=np.float32)
    sizes, offs, off = {}, {}, 0

    def take(name, n):
        nonlocal off
        sizes[name] = n; offs[name] = off; off += n

    for k, n in [('W0', C_HID * C_IN * 9), ('b0', C_HID), ('bn0g', C_HID), ('bn0b', C_HID),
                 ('bn0m', C_HID), ('bn0v', C_HID),
                 ('Wq', D_MODEL * D_MODEL), ('Wk', D_MODEL * D_MODEL),
                 ('Wv', D_MODEL * D_MODEL), ('Wo', D_MODEL * D_MODEL),
                 ('Wff1', D_MODEL * D_FF), ('bff1', D_FF), ('Wff2', D_FF * D_MODEL), ('bff2', D_MODEL),
                 ('ln1g', D_MODEL), ('ln1b', D_MODEL), ('ln2g', D_MODEL), ('ln2b', D_MODEL),
                 ('Wp1', 32 * C_HID * 9), ('bp1', 32), ('Wp2', PCH_LEGACY * 32), ('bp2', PCH_LEGACY),
                 ('Wv1', 32 * C_HID * 9), ('bv1', 32), ('Wl1', 256 * 3200), ('bl1', 256),
                 ('Wl2', 256), ('bl2', 1)]:
        take(k, n)
    for i in range(RES_BLOCKS * 2): take(f'Wr{i}', C_HID * C_HID * 9)
    for i in range(RES_BLOCKS * 2): take(f'br{i}', C_HID)
    for bn in ['bng', 'bnb', 'bnm', 'bnv']:
        for i in range(RES_BLOCKS * 2): take(f'{bn}{i}', C_HID)
    legacy = off
    v2 = v3 = False
    if flat.size == legacy:
        pass
    elif flat.size == legacy + V2_TAIL:
        v2 = True
    elif flat.size == legacy + V2_TAIL + V3_TAIL:
        v2 = v3 = True
    else:
        raise ValueError(f'weights size mismatch: {flat.size} floats '
                         f'(v1={legacy} / v2={legacy + V2_TAIL} / v3={legacy + V2_TAIL + V3_TAIL})')
    if v2:
        for i in range(RES_BLOCKS): take(f'grn{i}', C_HID)
        for k in ['WqM', 'WkM', 'WvM', 'WoM']: take(k, D_MODEL * D_MODEL)
        take('Dw', C_HID * C_HID * 2 * 2); take('Db', C_HID)
        take('Uw', C_HID * C_HID * 2 * 2); take('Ub', C_HID)
        take('ln_mg', D_MODEL); take('ln_mb', D_MODEL)
        take('rpbT', HEADS * 361)
    if v3:
        take('Wp2x', (PCH - PCH_LEGACY) * 32); take('bp2x', PCH - PCH_LEGACY)
        take('plg', PLEVEL_TAIL // 2); take('plb', PLEVEL_TAIL // 2)
        for i in range(N_ATTN_EXTRA):
            for k, n in ATT_K_TAIL:
                take(f'attnX{i}_{k}', n)
        take('flags', FLAG_N)
    out = {k: flat[offs[k]:offs[k] + sizes[k]] for k in sizes}
    if not v3:
        _warm_init_policy(out)
    out['__v2__'] = v2
    out['__v3__'] = v3
    assert flat.size == off, f'layout drift: consumed {off} of {flat.size}'
    return out


class FrozenBN(nn.Module):
    """v1 遗留的归一化张量（g,b,m,v）。v3 训练侧把 m,v 当作 running statistics 更新，
    推理侧仍是仿射 (x-m)/sqrt(v+1e-5)*g+b —— 运行时（JS/GPU/OV）零改动。"""

    def __init__(self, g, b, m, v, norm_mode=0, momentum=0.1):
        super().__init__()
        r = lambda a: torch.from_numpy(a.copy()).reshape(1, -1, 1, 1)
        self.g = nn.Parameter(r(g)); self.b = nn.Parameter(r(b))
        self.m = nn.Parameter(r(m)); self.v = nn.Parameter(r(v))
        self.norm_mode = norm_mode      # 1 = 训练时用批统计并把 running stats 写回 m,v（真 BN）
        self.momentum = momentum

    def forward(self, x):
        if self.training and self.norm_mode == 1:
            bm = x.mean(dim=(0, 2, 3), keepdim=True)
            bv = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            with torch.no_grad():       # running stats（推理侧仍用 m,v 的仿射，运行时零改动）
                self.m.data.mul_(1 - self.momentum).add_(bm * self.momentum)
                self.v.data.mul_(1 - self.momentum).add_(bv * self.momentum)
            return (x - bm) / torch.sqrt(bv + 1e-5) * self.g + self.b
        return (x - self.m) / torch.sqrt(self.v + 1e-5) * self.g + self.b


# GRN（ConvNeXt V2 残差式）：X_out = X + γ ⊙ X/(‖X_c‖_spatial+ε)；γ 零初始化 → 恒等
class GRN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        gx = x.pow(2).sum(dim=(2, 3), keepdim=True).sqrt()
        return x + self.gamma.view(1, -1, 1, 1) * x / (gx + 1e-6)


class ResBlock(nn.Module):
    def __init__(self, W, idx):
        super().__init__()
        self.conv1 = nn.Conv2d(C_HID, C_HID, 3, padding=1, bias=True)
        self.conv1.weight.data = torch.from_numpy(W[f'Wr{idx}'].reshape(C_HID, C_HID, 3, 3).copy())
        self.conv1.bias.data = torch.from_numpy(W[f'br{idx}'].copy())
        self.bn1 = FrozenBN(W[f'bng{idx}'], W[f'bnb{idx}'], W[f'bnm{idx}'], W[f'bnv{idx}'])
        self.conv2 = nn.Conv2d(C_HID, C_HID, 3, padding=1, bias=True)
        self.conv2.weight.data = torch.from_numpy(W[f'Wr{idx+1}'].reshape(C_HID, C_HID, 3, 3).copy())
        self.conv2.bias.data = torch.from_numpy(W[f'br{idx+1}'].copy())
        self.bn2 = FrozenBN(W[f'bng{idx+1}'], W[f'bnb{idx+1}'], W[f'bnm{idx+1}'], W[f'bnv{idx+1}'])
        self.grn = GRN(C_HID)

    def forward(self, f):
        h = F.relu(self.bn1(self.conv1(f)))
        h = F.relu(self.bn2(self.conv2(h)))
        return self.grn(torch.relu(f + h))


# MANO（Multipole Attention, arXiv 2507.02748）：
#   各级共享 D(k2s2) 降采样 → 共享 QKV/LN 的窗口注意力(窗5) → 共享 U(k2s2) 上采样 → 求和 → 残差
#   恒等迁移：Wo_m 与 Ub/Db 零初始化 → 分支输出严格为 0
#   v3 按级仿射：共享 LN 之后逐级 γ/β（默认 1/0 → 恒等；由 flags[plevel] 决定是否训练）
class MANO(nn.Module):
    def __init__(self, W=None):
        super().__init__()
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.down = nn.Conv2d(C_HID, C_HID, 2, stride=2)
        self.up = nn.ConvTranspose2d(C_HID, C_HID, 2, stride=2)
        self.ln = nn.LayerNorm(D_MODEL)
        self.plg = nn.Parameter(torch.ones(MANO_LEVELS, D_MODEL))
        self.plb = nn.Parameter(torch.zeros(MANO_LEVELS, D_MODEL))
        # MANO 窗口开关（v3 flags[window]）：0 = L0 也做全图注意力（消融臂用）
        self.window_on = True
        if W is not None and W.get('__v3__'):
            self.window_on = bool(W['flags'][FLAG_IDX['window']] > 0.5)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
        if W is not None and W.get('__v2__'):
            self.Wq.weight.data = torch.from_numpy(W['WqM'].reshape(D_MODEL, D_MODEL).copy())
            self.Wk.weight.data = torch.from_numpy(W['WkM'].reshape(D_MODEL, D_MODEL).copy())
            self.Wv.weight.data = torch.from_numpy(W['WvM'].reshape(D_MODEL, D_MODEL).copy())
            self.Wo.weight.data = torch.from_numpy(W['WoM'].reshape(D_MODEL, D_MODEL).copy())
            self.down.weight.data = torch.from_numpy(W['Dw'].reshape(C_HID, C_HID, 2, 2).copy())
            self.down.bias.data = torch.from_numpy(W['Db'].copy())
            self.up.weight.data = torch.from_numpy(W['Uw'].reshape(C_HID, C_HID, 2, 2).copy())
            self.up.bias.data = torch.from_numpy(W['Ub'].copy())
            self.ln.weight.data = torch.from_numpy(W['ln_mg'].copy())
            self.ln.bias.data = torch.from_numpy(W['ln_mb'].copy())
        if W is not None and W.get('__v3__'):
            self.plg.data = torch.from_numpy(W['plg'].reshape(MANO_LEVELS, D_MODEL).copy())
            self.plb.data = torch.from_numpy(W['plb'].reshape(MANO_LEVELS, D_MODEL).copy())

    def _window_attn(self, x, lv):
        B, C, H, Wd = x.shape
        w = MANO_WINDOW
        t = x.flatten(2).transpose(1, 2)                 # (B, HW, C)
        t = self.ln(t)
        t = t * self.plg[lv] + self.plb[lv]              # 按级仿射（恒等初始化）
        q, k, v = self.Wq(t), self.Wk(t), self.Wv(t)
        if self.window_on and H > w and Wd > w and H % w == 0 and Wd % w == 0:
            gh, gw = H // w, Wd // w

            def win(u):
                u2 = u.view(B, gh, w, gw, w, D_MODEL)
                return u2.permute(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, w * w, D_MODEL)
            qw, kw, vw = win(q), win(k), win(v)
            nw = gh * gw
        else:
            qw, kw, vw = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)   # 单窗=全图
            nw = 1
        qh = qw.view(B, nw, -1, HEADS, HD).transpose(2, 3)
        kh = kw.view(B, nw, -1, HEADS, HD).transpose(2, 3)
        vh = vw.view(B, nw, -1, HEADS, HD).transpose(2, 3)
        att = torch.softmax(qh @ kh.transpose(-2, -1) / (HD ** 0.5), dim=-1)
        o = (att @ vh).transpose(2, 3).reshape(B, nw, -1, D_MODEL)
        o = self.Wo(o)
        if nw > 1:
            gh, gw = H // w, Wd // w
            o = o.view(B, gh, gw, w, w, D_MODEL).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, Wd)
        else:
            o = o.squeeze(1).transpose(1, 2).reshape(B, C, H, Wd)
        return o

    def forward(self, x):
        B, C, H0, W0 = x.shape
        sizes = [(H0, W0)]
        h = x
        for _ in range(1, MANO_LEVELS):
            h = self.down(h)
            sizes.append((h.shape[2], h.shape[3]))
        total = torch.zeros_like(x)
        h = x
        for lv in range(MANO_LEVELS):
            if lv > 0:
                h = self.down(h)
            u = self._window_attn(h, lv)
            for m in range(lv):                     # 逐级上采样，目标=降采样链同尺寸
                u = self.up(u, output_size=sizes[lv - 1 - m])
            total = total + u
        return x + total


class ExtraAttn(nn.Module):
    """v3 额外末段注意力层：零初始化**残差式**块
        h = t + Wo(attn(LN1(t))) ;  h = h + Wff2(relu(Wff1(LN2(h))))
     Wo 与 Wff2 零初始化 → 前向严格恒等（现有 pre-LN 块无法零初始化恒等，故新增层用此形式）。"""

    def __init__(self, W=None, idx=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.ln2 = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wff1 = nn.Linear(D_MODEL, D_FF)
        self.Wff2 = nn.Linear(D_FF, D_MODEL)
        for m in (self.Wq, self.Wk, self.Wv, self.Wo, self.Wff1, self.Wff2):
            m.reset_parameters()               # 内部小随机（保证 Wo 有非零梯度）
        nn.init.zeros_(self.Wo.weight)         # 仅输出投影置零 → 前向恒等
        nn.init.zeros_(self.Wff1.bias); nn.init.zeros_(self.Wff2.weight); nn.init.zeros_(self.Wff2.bias)
        if W is not None and W.get('__v3__'):
            p = f'attnX{idx}_'
            self.Wq.weight.data = torch.from_numpy(W[p + 'Wq'].reshape(D_MODEL, D_MODEL).T.copy())
            self.Wk.weight.data = torch.from_numpy(W[p + 'Wk'].reshape(D_MODEL, D_MODEL).T.copy())
            self.Wv.weight.data = torch.from_numpy(W[p + 'Wv'].reshape(D_MODEL, D_MODEL).T.copy())
            self.Wo.weight.data = torch.from_numpy(W[p + 'Wo'].reshape(D_MODEL, D_MODEL).T.copy())
            self.Wff1.weight.data = torch.from_numpy(W[p + 'Wff1'].reshape(D_MODEL, D_FF).T.copy())
            self.Wff1.bias.data = torch.from_numpy(W[p + 'bff1'].copy())
            self.Wff2.weight.data = torch.from_numpy(W[p + 'Wff2'].reshape(D_FF, D_MODEL).T.copy())
            self.Wff2.bias.data = torch.from_numpy(W[p + 'bff2'].copy())
            self.ln1.weight.data = torch.from_numpy(W[p + 'ln1g'].copy())
            self.ln1.bias.data = torch.from_numpy(W[p + 'ln1b'].copy())
            self.ln2.weight.data = torch.from_numpy(W[p + 'ln2g'].copy())
            self.ln2.bias.data = torch.from_numpy(W[p + 'ln2b'].copy())

    def forward(self, t):
        B = t.shape[0]
        h1 = self.ln1(t)
        q = self.Wq(h1).view(B, N_POS, HEADS, HD).transpose(1, 2)
        k = self.Wk(h1).view(B, N_POS, HEADS, HD).transpose(1, 2)
        v = self.Wv(h1).view(B, N_POS, HEADS, HD).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (HD ** 0.5), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, N_POS, D_MODEL)
        h = t + self.Wo(o)
        h = h + self.Wff2(F.relu(self.Wff1(self.ln2(h))))
        return h


class Net(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.conv0 = nn.Conv2d(C_IN, C_HID, 3, padding=1, bias=True)
        self.conv0.weight.data = torch.from_numpy(W['W0'].reshape(C_HID, C_IN, 3, 3).copy())
        self.conv0.bias.data = torch.from_numpy(W['b0'].copy())
        self.bn0 = FrozenBN(W['bn0g'], W['bn0b'], W['bn0m'], W['bn0v'])
        self.blocks = nn.ModuleList([ResBlock(W, i * 2) for i in range(RES_BLOCKS)])
        self.mano = MANO(W)
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False); self.Wq.weight.data = torch.from_numpy(W['Wq'].reshape(D_MODEL, D_MODEL).T.copy())
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False); self.Wk.weight.data = torch.from_numpy(W['Wk'].reshape(D_MODEL, D_MODEL).T.copy())
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False); self.Wv.weight.data = torch.from_numpy(W['Wv'].reshape(D_MODEL, D_MODEL).T.copy())
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False); self.Wo.weight.data = torch.from_numpy(W['Wo'].reshape(D_MODEL, D_MODEL).T.copy())
        self.ln1 = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.ln1.weight.data = torch.from_numpy(W['ln1g'].copy()); self.ln1.bias.data = torch.from_numpy(W['ln1b'].copy())
        self.ff1 = nn.Linear(D_MODEL, D_FF); self.ff1.weight.data = torch.from_numpy(W['Wff1'].reshape(D_MODEL, D_FF).T.copy()); self.ff1.bias.data = torch.from_numpy(W['bff1'].copy())
        self.ff2 = nn.Linear(D_FF, D_MODEL); self.ff2.weight.data = torch.from_numpy(W['Wff2'].reshape(D_FF, D_MODEL).T.copy()); self.ff2.bias.data = torch.from_numpy(W['bff2'].copy())
        self.ln2 = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.ln2.weight.data = torch.from_numpy(W['ln2g'].copy()); self.ln2.bias.data = torch.from_numpy(W['ln2b'].copy())
        # v3 额外末段注意力层（前 n_attn_extra 层参与前向；默认 0 → 与 v2 逐位一致）
        self.attnX = nn.ModuleList([ExtraAttn(W, i) for i in range(N_ATTN_EXTRA)])
        self.n_attn_extra = int(W['flags'][FLAG_IDX['attn_extra']]) if W.get('__v3__') else 0
        self.n_attn_extra = max(0, min(N_ATTN_EXTRA, self.n_attn_extra))
        # 2D 相对位置偏置（末段标准 T）：rpb[h, dr+9, dc+9]，零初始化 → 恒等
        self.rpb = nn.Parameter(torch.zeros(HEADS, 19, 19))
        if W.get('__v2__'):
            self.rpb.data = torch.from_numpy(W['rpbT'].reshape(HEADS, 19, 19).copy())
        pos = torch.arange(10)
        r_ = pos.unsqueeze(1).expand(10, 10).reshape(100)
        c_ = pos.unsqueeze(0).expand(10, 10).reshape(100)
        self.register_buffer('idx_r', ((r_[:, None] - r_[None, :]) + 9).long())
        self.register_buffer('idx_c', ((c_[:, None] - c_[None, :]) + 9).long())
        # policy 头：v3 = 旧 100 行（Wp2/bp2）+ 新 60 行（Wp2x/bp2x）；
        # v1/v2 权重的新 60 行由 _warm_init_policy 语义化 warm-init 补齐（旧 100 行逐位不变）
        Wp2 = np.concatenate([W['Wp2'].reshape(PCH_LEGACY, 32),
                              W['Wp2x'].reshape(PCH - PCH_LEGACY, 32)])
        bp2 = np.concatenate([W['bp2'], W['bp2x']])
        self.polc = nn.Conv2d(C_HID, 32, 3, padding=1, bias=True)
        self.polc.weight.data = torch.from_numpy(W['Wp1'].reshape(32, C_HID, 3, 3).copy()); self.polc.bias.data = torch.from_numpy(W['bp1'].copy())
        self.polo = nn.Conv2d(32, PCH, 1, bias=True)
        self.polo.weight.data = torch.from_numpy(Wp2.reshape(PCH, 32, 1, 1).copy())
        self.polo.bias.data = torch.from_numpy(bp2.copy())
        self.valc = nn.Conv2d(C_HID, 32, 3, padding=1, bias=True)
        self.valc.weight.data = torch.from_numpy(W['Wv1'].reshape(32, C_HID, 3, 3).copy()); self.valc.bias.data = torch.from_numpy(W['bv1'].copy())
        self.vall1 = nn.Linear(32 * N_POS, 256); self.vall1.weight.data = torch.from_numpy(W['Wl1'].reshape(256, 3200).copy()); self.vall1.bias.data = torch.from_numpy(W['bl1'].copy())
        self.vall2 = nn.Linear(256, 1); self.vall2.weight.data = torch.from_numpy(W['Wl2'].reshape(1, 256).copy()); self.vall2.bias.data = torch.from_numpy(W['bl2'].copy())
        # 归一化模式：flags[norm]=1 → m,v 作为 running statistics（训练侧），此处仅记录
        self.norm_mode = int(W['flags'][FLAG_IDX['norm']]) if W.get('__v3__') else 0
        self.mano_plevel_trainable = bool(W.get('__v3__') and W['flags'][FLAG_IDX['plevel']] > 0.5)

    def trunk(self, x):
        f = F.relu(self.bn0(self.conv0(x)))
        for blk in self.blocks[:3]:
            f = blk(f)
        f = self.mano(f)
        for blk in self.blocks[3:]:
            f = blk(f)
        return f

    def forward(self, x):
        f = self.trunk(x)
        t = f.flatten(2).transpose(1, 2)
        q, k, v = self.Wq(t), self.Wk(t), self.Wv(t)
        q = q.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        k = k.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        v = v.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (HD ** 0.5)
        scores = scores + self.rpb[:, self.idx_r, self.idx_c].unsqueeze(0)
        att = torch.softmax(scores, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(-1, N_POS, D_MODEL)
        h = self.ln1(self.Wo(o) + t)
        h2 = self.ln2(self.ff2(F.relu(self.ff1(h))) + h)
        for i in range(self.n_attn_extra):
            h2 = self.attnX[i](h2)
        feat = h2.transpose(1, 2).reshape(-1, C_HID, 10, 10)
        pol = self.polo(F.relu(self.polc(feat))).flatten(1)
        hv = F.relu(self.valc(feat)).flatten(1)
        raw = self.vall2(F.relu(self.vall1(hv))).squeeze(-1)
        return pol, raw


class Head(nn.Module):
    """主干解冻包装：全参数 requires_grad=True（兼容 import）。"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        return self.net(x)


def build_net(weights_path):
    W = load_weights_bin(weights_path)
    net = Net(W).eval()
    # 按级仿射：关闭时冻结在恒等（不参与训练）
    if not net.mano_plevel_trainable:
        net.mano.plg.requires_grad_(False)
        net.mano.plb.requires_grad_(False)
    return net


# 导出 weights.bin（布局与 cnn.js saveWeights 严格一致 + v2 尾部 + v3 尾部）
def save_weights_bin(net, path):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = {}
    out['W0'] = sd['conv0.weight'].flatten(); out['b0'] = sd['conv0.bias']
    out['bn0g'] = sd['bn0.g']; out['bn0b'] = sd['bn0.b']; out['bn0m'] = sd['bn0.m']; out['bn0v'] = sd['bn0.v']
    for key in ['Wq', 'Wk', 'Wv', 'Wo']: out[key] = sd[f'{key}.weight'].T.flatten()
    out['Wff1'] = sd['ff1.weight'].T.flatten(); out['bff1'] = sd['ff1.bias']
    out['Wff2'] = sd['ff2.weight'].T.flatten(); out['bff2'] = sd['ff2.bias']
    out['ln1g'] = sd['ln1.weight']; out['ln1b'] = sd['ln1.bias']
    out['ln2g'] = sd['ln2.weight']; out['ln2b'] = sd['ln2.bias']
    out['Wp1'] = sd['polc.weight'].flatten(); out['bp1'] = sd['polc.bias']
    # policy 头拆回 legacy(100 行) + Wp2x(60 行)
    Wp2 = sd['polo.weight'].reshape(PCH, 32)
    out['Wp2'] = Wp2[:PCH_LEGACY].flatten(); out['bp2'] = sd['polo.bias'][:PCH_LEGACY]
    out['Wp2x'] = Wp2[PCH_LEGACY:].flatten(); out['bp2x'] = sd['polo.bias'][PCH_LEGACY:]
    out['Wv1'] = sd['valc.weight'].flatten(); out['bv1'] = sd['valc.bias']
    out['Wl1'] = sd['vall1.weight'].flatten(); out['bl1'] = sd['vall1.bias']
    out['Wl2'] = sd['vall2.weight'].flatten(); out['bl2'] = sd['vall2.bias']
    W_KEYS = ['W0', 'b0', 'bn0g', 'bn0b', 'bn0m', 'bn0v', 'Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2',
              'ln1g', 'ln1b', 'ln2g', 'ln2b', 'Wp1', 'bp1', 'Wp2', 'bp2',
              'Wv1', 'bv1', 'Wl1', 'bl1', 'Wl2', 'bl2']
    Wr, br, bng, bnb, bnm, bnv = [], [], [], [], [], []
    for b in range(RES_BLOCKS):
        Wr += [sd[f'blocks.{b}.conv1.weight'].flatten(), sd[f'blocks.{b}.conv2.weight'].flatten()]
        br += [sd[f'blocks.{b}.conv1.bias'], sd[f'blocks.{b}.conv2.bias']]
        bng += [sd[f'blocks.{b}.bn1.g'], sd[f'blocks.{b}.bn2.g']]
        bnb += [sd[f'blocks.{b}.bn1.b'], sd[f'blocks.{b}.bn2.b']]
        bnm += [sd[f'blocks.{b}.bn1.m'], sd[f'blocks.{b}.bn2.m']]
        bnv += [sd[f'blocks.{b}.bn1.v'], sd[f'blocks.{b}.bn2.v']]
    legacy = [out[k] for k in W_KEYS] + Wr + br + bng + bnb + bnm + bnv
    tail = [sd[f'blocks.{i}.grn.gamma'].flatten() for i in range(RES_BLOCKS)] + [
        sd['mano.Wq.weight'].flatten(), sd['mano.Wk.weight'].flatten(),
        sd['mano.Wv.weight'].flatten(), sd['mano.Wo.weight'].flatten(),
        sd['mano.down.weight'].flatten(), sd['mano.down.bias'].flatten(),
        sd['mano.up.weight'].flatten(), sd['mano.up.bias'].flatten(),
        sd['mano.ln.weight'].flatten(), sd['mano.ln.bias'].flatten(),
        sd['rpb'].flatten(),
    ]
    v3 = [out['Wp2x'].flatten(), out['bp2x'].flatten(),
          sd['mano.plg'].flatten(), sd['mano.plb'].flatten()]
    for i in range(N_ATTN_EXTRA):
        p = f'attnX.{i}.'
        v3 += [sd[p + 'Wq.weight'].T.flatten(), sd[p + 'Wk.weight'].T.flatten(),
               sd[p + 'Wv.weight'].T.flatten(), sd[p + 'Wo.weight'].T.flatten(),
               sd[p + 'Wff1.weight'].T.flatten(), sd[p + 'Wff1.bias'],
               sd[p + 'Wff2.weight'].T.flatten(), sd[p + 'Wff2.bias'],
               sd[p + 'ln1.weight'], sd[p + 'ln1.bias'],
               sd[p + 'ln2.weight'], sd[p + 'ln2.bias']]
    v3 += [net_flags(net)]
    flat = np.concatenate([p.astype(np.float32).flatten() for p in legacy + tail + v3])
    if flat.size != V3_FLOATS:
        raise ValueError(f'v3 export size mismatch: {flat.size} != {V3_FLOATS}')
    flat.tofile(path)
    return flat.size


def net_flags(net):
    f = default_flags()
    f[FLAG_IDX['plevel']] = 1.0 if net.mano_plevel_trainable else 0.0
    f[FLAG_IDX['window']] = 1.0 if net.mano.window_on else 0.0
    f[FLAG_IDX['attn_extra']] = float(net.n_attn_extra)
    f[FLAG_IDX['norm']] = float(getattr(net, 'norm_mode', 0))
    return f
