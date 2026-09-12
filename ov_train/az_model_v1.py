# -*- coding: utf-8 -*-
# 共享模型定义 + weights.bin 加载（布局与 cnn.js saveWeights 严格一致）
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SERVER = 'D:/data/新建文件夹/chess_game/server'
C_IN, C_HID, RES_BLOCKS, HEADS, D_MODEL, D_FF, PCH, N_POS = 24, 128, 6, 4, 128, 256, 100, 100
HD = D_MODEL // HEADS

def load_weights_bin(path):
    flat = np.fromfile(path, dtype=np.float32)
    sizes, offs, off = {}, {}, 0
    def take(name, n):
        nonlocal off
        sizes[name] = n; offs[name] = off; off += n
    for k, n in [('W0',C_HID*C_IN*9),('b0',C_HID),('bn0g',C_HID),('bn0b',C_HID),('bn0m',C_HID),('bn0v',C_HID),
                 ('Wq',D_MODEL*D_MODEL),('Wk',D_MODEL*D_MODEL),('Wv',D_MODEL*D_MODEL),('Wo',D_MODEL*D_MODEL),
                 ('Wff1',D_MODEL*D_FF),('bff1',D_FF),('Wff2',D_FF*D_MODEL),('bff2',D_MODEL),
                 ('ln1g',D_MODEL),('ln1b',D_MODEL),('ln2g',D_MODEL),('ln2b',D_MODEL),
                 ('Wp1',32*C_HID*9),('bp1',32),('Wp2',PCH*32),('bp2',PCH),
                 ('Wv1',32*C_HID*9),('bv1',32),('Wl1',256*3200),('bl1',256),('Wl2',256),('bl2',1)]:
        take(k, n)
    for i in range(RES_BLOCKS * 2): take(f'Wr{i}', C_HID*C_HID*9)
    for i in range(RES_BLOCKS * 2): take(f'br{i}', C_HID)
    for bn in ['bng','bnb','bnm','bnv']:
        for i in range(RES_BLOCKS * 2): take(f'{bn}{i}', C_HID)
    if off != flat.size: raise ValueError(f'weights size mismatch: {off} != {flat.size}')
    return {k: flat[offs[k]:offs[k]+sizes[k]] for k in sizes}

class FrozenBN(nn.Module):
    def __init__(self, g, b, m, v):
        super().__init__()
        r = lambda a: torch.from_numpy(a.copy()).reshape(1, -1, 1, 1)
        self.g = nn.Parameter(r(g)); self.b = nn.Parameter(r(b))
        self.m = nn.Parameter(r(m)); self.v = nn.Parameter(r(v))
    def forward(self, x):
        return (x - self.m) / torch.sqrt(self.v + 1e-5) * self.g + self.b

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
    def forward(self, f):
        h = F.relu(self.bn1(self.conv1(f)))
        h = F.relu(self.bn2(self.conv2(h)))
        return torch.relu(f + h)

class Net(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.conv0 = nn.Conv2d(C_IN, C_HID, 3, padding=1, bias=True)
        self.conv0.weight.data = torch.from_numpy(W['W0'].reshape(C_HID, C_IN, 3, 3).copy())
        self.conv0.bias.data = torch.from_numpy(W['b0'].copy())
        self.bn0 = FrozenBN(W['bn0g'], W['bn0b'], W['bn0m'], W['bn0v'])
        self.blocks = nn.ModuleList([ResBlock(W, i * 2) for i in range(RES_BLOCKS)])
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
        self.polc = nn.Conv2d(C_HID, 32, 3, padding=1, bias=True)
        self.polc.weight.data = torch.from_numpy(W['Wp1'].reshape(32, C_HID, 3, 3).copy()); self.polc.bias.data = torch.from_numpy(W['bp1'].copy())
        self.polo = nn.Conv2d(32, PCH, 1, bias=True)
        self.polo.weight.data = torch.from_numpy(W['Wp2'].reshape(PCH, 32, 1, 1).copy()); self.polo.bias.data = torch.from_numpy(W['bp2'].copy())
        self.valc = nn.Conv2d(C_HID, 32, 3, padding=1, bias=True)
        self.valc.weight.data = torch.from_numpy(W['Wv1'].reshape(32, C_HID, 3, 3).copy()); self.valc.bias.data = torch.from_numpy(W['bv1'].copy())
        self.vall1 = nn.Linear(32 * N_POS, 256); self.vall1.weight.data = torch.from_numpy(W['Wl1'].reshape(256, 3200).copy()); self.vall1.bias.data = torch.from_numpy(W['bl1'].copy())
        self.vall2 = nn.Linear(256, 1); self.vall2.weight.data = torch.from_numpy(W['Wl2'].reshape(1, 256).copy()); self.vall2.bias.data = torch.from_numpy(W['bl2'].copy())

    def trunk(self, x):
        f = F.relu(self.bn0(self.conv0(x)))
        for blk in self.blocks: f = blk(f)
        return f

    def forward(self, x):
        f = self.trunk(x)
        t = f.flatten(2).transpose(1, 2)
        q, k, v = self.Wq(t), self.Wk(t), self.Wv(t)
        q = q.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        k = k.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        v = v.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (HD ** 0.5), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(-1, N_POS, D_MODEL)
        h = self.ln1(self.Wo(o) + t)
        h2 = self.ln2(self.ff2(F.relu(self.ff1(h))) + h)
        feat = h2.transpose(1, 2).reshape(-1, C_HID, 10, 10)
        pol = self.polo(F.relu(self.polc(feat))).flatten(1)
        hv = F.relu(self.valc(feat)).flatten(1)
        raw = self.vall2(F.relu(self.vall1(hv))).squeeze(-1)
        return pol, raw

class Head(nn.Module):
    """主干解冻包装：全参数训练。把整网所有参数 requires_grad 置 True，
    forward 直接透传 net（训练器实例化后仍用 net 前向/保存，此类只为解冻副作用+兼容 import）。"""
    def __init__(self, net):
        super().__init__()
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(True)
    def forward(self, x):
        return self.net(x)

def build_net(weights_path):
    return Net(load_weights_bin(weights_path)).eval()

# 导出 weights.bin（与 cnn.js saveWeights 布局一致）
def save_weights_bin(net, path):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = {}
    out['W0'] = sd['conv0.weight'].flatten(); out['b0'] = sd['conv0.bias']
    out['bn0g'] = sd['bn0.g']; out['bn0b'] = sd['bn0.b']; out['bn0m'] = sd['bn0.m']; out['bn0v'] = sd['bn0.v']
    for key in ['Wq','Wk','Wv','Wo']: out[key] = sd[f'{key}.weight'].T.flatten()
    out['Wff1'] = sd['ff1.weight'].T.flatten(); out['bff1'] = sd['ff1.bias']
    out['Wff2'] = sd['ff2.weight'].T.flatten(); out['bff2'] = sd['ff2.bias']
    out['ln1g'] = sd['ln1.weight']; out['ln1b'] = sd['ln1.bias']
    out['ln2g'] = sd['ln2.weight']; out['ln2b'] = sd['ln2.bias']
    out['Wp1'] = sd['polc.weight'].flatten(); out['bp1'] = sd['polc.bias']
    out['Wp2'] = sd['polo.weight'].flatten(); out['bp2'] = sd['polo.bias']
    out['Wv1'] = sd['valc.weight'].flatten(); out['bv1'] = sd['valc.bias']
    out['Wl1'] = sd['vall1.weight'].flatten(); out['bl1'] = sd['vall1.bias']
    out['Wl2'] = sd['vall2.weight'].flatten(); out['bl2'] = sd['vall2.bias']
    W_KEYS = ['W0','b0','bn0g','bn0b','bn0m','bn0v','Wq','Wk','Wv','Wo','Wff1','bff1','Wff2','bff2',
              'ln1g','ln1b','ln2g','ln2b','Wp1','bp1','Wp2','bp2','Wv1','bv1','Wl1','bl1','Wl2','bl2']
    Wr, br, bng, bnb, bnm, bnv = [], [], [], [], [], []
    for b in range(RES_BLOCKS):
        Wr += [sd[f'blocks.{b}.conv1.weight'].flatten(), sd[f'blocks.{b}.conv2.weight'].flatten()]
        br += [sd[f'blocks.{b}.conv1.bias'], sd[f'blocks.{b}.conv2.bias']]
        bng += [sd[f'blocks.{b}.bn1.g'], sd[f'blocks.{b}.bn2.g']]
        bnb += [sd[f'blocks.{b}.bn1.b'], sd[f'blocks.{b}.bn2.b']]
        bnm += [sd[f'blocks.{b}.bn1.m'], sd[f'blocks.{b}.bn2.m']]
        bnv += [sd[f'blocks.{b}.bn1.v'], sd[f'blocks.{b}.bn2.v']]
    flat = np.concatenate([p.astype(np.float32).flatten() for p in
        [out[k] for k in W_KEYS] + Wr + br + bng + bnb + bnm + bnv])
    flat.tofile(path)
    return flat.size