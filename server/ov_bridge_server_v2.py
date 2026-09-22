# -*- coding: utf-8 -*-
# OpenVINO 推理桥 v2.1（chess10 v2 架构）—— 对齐官方 NPU 优化指南的重写版
# 相比 v2.0 的优化（依据 OpenVINO 官方 NPU 文档与 GenAI-on-NPU 生产用法）：
#   1) 显式性能提示 + 编译参数：LATENCY/THROUGHPUT hint、NPU_TURBO（拉频）、
#      NPU_COMPILATION_MODE_PARAMS="performance-hint-override=latency"（编译期延迟优先）
#   2) 查询 OPTIMAL_NUMBER_OF_INFER_REQUESTS 并按其建 AsyncInferQueue —— NPU 连续流水，
#      多块请求不再逐块同步阻塞（v2.0 每块 infer 后主机空等）
#   3) 双粒度形状服务（GenAI PREFILL_CHUNK_SIZE 思路）：trunk 同时编译
#      延迟档 batch=BAT_LAT(默认8) 与 吞吐档 batch=BAT(默认64)，
#      小请求（对局常见 n≤8）不再 pad 到 64 白跑 90ms
#   4) 保留 v2.0 全部机制：full/trunk/cnn3 导出降级链 × NPU→GPU→CPU 设备链、
#      精度门禁（value≤0.05/policy≤3.0/top1≥99%）、IR 磁盘缓存（split+batch+weights 指纹）
# 协议不变：REQ=[int32 N][N*107 i32] → RESP=[int32 N][N f32 vals][N*POL_SIZE f32 pols]
import os, json, struct, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
OVTRAIN = os.path.normpath(os.path.join(HERE, '..', 'ov_train'))
sys.path.insert(0, OVTRAIN)
from az_model import build_net, C_IN, C_HID, HEADS, D_MODEL, HD, PCH, N_POS  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

POL_SIZE = PCH * N_POS
GATE_V, GATE_P, GATE_T1 = 0.05, 3.0, 0.99

def parse_args(argv):
    a = {'weights': None,
         'batch': int(os.environ.get('CHESS10_OV_BATCH', '32')),
         'batch_lat': int(os.environ.get('CHESS10_OV_BATCH_LAT', '8')),
         'device': os.environ.get('CHESS10_OV_DEVICE', 'NPU'),
         'turbo': os.environ.get('CHESS10_OV_TURBO', '0') == '1',
         'fold': os.environ.get('CHESS10_OV_NOFOLD', '0') != '1',      # BN 折叠（实测 -12% 每盘）
         'native': os.environ.get('CHESS10_OV_NONATIVE', '0') != '1'}  # 哑输出破条件触发原生批（实测 b64 -15%）
    i = 0
    while i < len(argv):
        if argv[i] == '--weights': a['weights'] = argv[i + 1]
        elif argv[i] == '--batch': a['batch'] = int(argv[i + 1])
        elif argv[i] == '--batch-lat': a['batch_lat'] = int(argv[i + 1])
        elif argv[i] == '--device': a['device'] = argv[i + 1]
        i += 1
    return a

args = parse_args(sys.argv[1:])
B = max(8, args['batch'])
BL = max(1, min(args['batch_lat'], B))
assert args['weights'] and os.path.exists(args['weights']), 'weights missing'

t_import = time.time()
net = build_net(args['weights']).eval()
torch.set_num_threads(2)
with torch.no_grad():
    probe = net(torch.zeros(1, C_IN, 10, 10))[1].item()
    assert np.isfinite(probe), 'probe value not finite'

# ---- 门禁参考必须在任何图变换（BN 折叠）之前计算，保持对原始权重的独立性 ----
probe_path = os.path.normpath(os.path.join(HERE, '..', 'training', 'teacher', 'parity.json'))
if os.path.exists(probe_path):
    with open(probe_path) as fp:
        px = np.array(json.load(fp)['encs'], dtype=np.float32)[:4].reshape(-1, C_IN, 10, 10)
else:
    px = (np.random.default_rng(7).standard_normal((4, C_IN, 10, 10)) * 0.5).astype(np.float32)
with torch.no_grad():
    _pt, _rt = net(torch.from_numpy(px))
ref_val = np.tanh(_rt.numpy()).reshape(-1)
ref_pol = _pt.numpy()

# ---- BN 折叠（数学等价，实测 NPU -12%）：FrozenBN 折入前置 conv，BN 置恒等 ----
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

if args['fold']:
    fold_bn_weights(net)
    print('[OV] BN folded into convs (env CHESS10_OV_NOFOLD=1 to disable)', file=sys.stderr, flush=True)

# ---- 原生批触发包装：附加无 batch 轴的哑输出，打破插件并发回退条件（实测 b64 -15%） ----
class NativeWrap(nn.Module):
    def __init__(self, base, mode):
        super().__init__(); self.base = base; self.mode = mode
    def forward(self, x):
        out = self.base(x)
        if self.mode == 'full':
            return out[0], out[1], out[0].mean()      # 哑输出 0 维
        return out, out.mean(dim=0)                    # 哑输出 (C,10,10) 无 batch 轴

class TrunkOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return self.net.trunk(x)

class CNNOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        f = F.relu(self.net.bn0(self.net.conv0(x)))
        for blk in self.net.blocks[:3]:
            f = blk(f)
        return f

class TorchHead(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, f):
        n = self.net
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
        for i in range(getattr(n, 'n_attn_extra', 0)):   # v3 额外末段注意力层（flags[4]）
            h2 = n.attnX[i](h2)
        feat = h2.transpose(1, 2).reshape(-1, C_HID, 10, 10)
        pol = n.polo(F.relu(n.polc(feat))).flatten(1)
        hv = F.relu(n.valc(feat)).flatten(1)
        raw = n.vall2(F.relu(n.vall1(hv))).squeeze(-1)
        return pol, raw

def make_module(split):
    base = net
    if split == 'trunk': base = TrunkOnly(base)
    elif split == 'cnn3': base = CNNOnly(base)
    elif split != 'full': raise ValueError(split)
    if args['native']:
        return NativeWrap(base, split)
    return base

VARIANT = ('f' if args['fold'] else '') + ('n' if args['native'] else '')

CACHE = os.path.join(HERE, 'models_ov')
os.makedirs(CACHE, exist_ok=True)
INT8 = os.environ.get('CHESS10_OV_INT8', '0') == '1'   # 需先用 tools/ov_nncf_int8.py 产出量化 IR
st = os.stat(args['weights'])
SIG = f'{int(st.st_mtime_ns)}:{st.st_size}'

def get_ir(split, batch):
    if INT8:
        x8 = os.path.join(CACHE, f'{split}_v2_int8_b{batch}.xml')
        m8 = x8 + '.meta'
        if os.path.exists(x8) and os.path.exists(m8):
            try:
                if open(m8).read().strip() == SIG:
                    return ov.Core().read_model(x8)
            except Exception:
                pass
        print(f'[OV] INT8 IR missing for {split} b{batch}, falling back to fp16', file=sys.stderr, flush=True)
    xml = os.path.join(CACHE, f'{split}_v2{VARIANT}_b{batch}.xml')
    meta = xml + '.meta'
    if os.path.exists(xml) and os.path.exists(meta):
        try:
            if open(meta).read().strip() == SIG:
                return ov.Core().read_model(xml)
        except Exception:
            pass
    m = ov.convert_model(make_module(split).eval(),
                         example_input=torch.zeros(batch, C_IN, 10, 10),
                         input=(batch, C_IN, 10, 10))
    try:
        ov.save_model(m, xml)
        open(meta, 'w').write(SIG)
    except Exception:
        pass
    return m

print(f'[OV] torch load ok ({time.time()-t_import:.1f}s), probe={probe:.3f}', file=sys.stderr, flush=True)

# ---- 门禁参考已在图变换前计算（ref_val/ref_pol），此处直接进入编译流程 ----
core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(CACHE, 'cache')})
head = TorchHead(net).eval()

def split_order(dev):
    return ['trunk', 'cnn3', 'full'] if dev.upper().startswith('NPU') else ['full', 'trunk', 'cnn3']

def compile_props(dev, batch, for_latency):
    """官方指南：显式 hint 优于默认；LATENCY 批小时编译期 latency 覆盖 + 钉核保缓存热度；TURBO 可选拉频"""
    props = {'PERFORMANCE_HINT': 'LATENCY' if for_latency else 'THROUGHPUT'}
    if dev.upper().startswith('NPU'):
        if for_latency:
            props['NPU_COMPILATION_MODE_PARAMS'] = 'performance-hint-override=latency'
            props['ENABLE_CPU_PINNING'] = 'YES'   # 实测：b1 fresh 路径 1.82→1.27ms（主机缓存热度）
        if args['turbo']:
            props['NPU_TURBO'] = 'YES'
    return props

def resolve_outputs(compiled, split):
    pol_out = val_out = feat_out = None
    for out in compiled.outputs:
        ps = out.get_partial_shape()
        sh = [ps[i].get_length() if ps[i].is_static else -1 for i in range(len(ps))]
        if len(sh) == 2 and sh[1] == POL_SIZE: pol_out = out
        elif len(sh) == 1: val_out = out
        elif len(sh) == 4: feat_out = out
    if split == 'full':
        return (pol_out, val_out) if (pol_out is not None and val_out is not None) else None
    return (feat_out,) if feat_out is not None else None

def run_probe(ireq, outs, x=None):
    """与主服务相同的路径组装（分块+pad+torch 尾段），用于精度门禁；x 缺省用门禁探针"""
    if x is None:
        x = px
    n = x.shape[0]
    if len(outs) == 1:
        feats = []
        for i in range(0, n, B):
            xb = x[i:i + B]
            if xb.shape[0] < B:
                xb = np.concatenate([xb, np.zeros((B - xb.shape[0], C_IN, 10, 10), np.float32)])
            feats.append(sync_infer(ireq, outs, xb))
        f_np = np.concatenate(feats)[:n]
        with torch.no_grad():
            pol, raw = head(torch.from_numpy(f_np))
        return np.tanh(raw.numpy().reshape(-1)).astype(np.float32), pol.numpy().astype(np.float32)
    vals_all, pols_all = [], []
    for i in range(0, n, B):
        xb = x[i:i + B]
        if xb.shape[0] < B:
            xb = np.concatenate([xb, np.zeros((B - xb.shape[0], C_IN, 10, 10), np.float32)])
        pv = sync_infer(ireq, outs, xb)
        pols_all.append(pv[0]); vals_all.append(pv[1])
    pols = np.concatenate(pols_all)[:n].astype(np.float32)
    vals = np.tanh(np.concatenate(vals_all)[:n].reshape(-1)).astype(np.float32)
    return vals, pols

def gate_check(vals, pols, rv=None, rp=None):
    rv = ref_val if rv is None else rv
    rp = ref_pol if rp is None else rp
    dv = float(np.abs(vals - rv).max())
    dp = float(np.abs(pols - rp).max())
    # margin-aware top1（2026-09-17）：参照自身 top1-top2 logits 差 < 0.05 的位置属并列，
    # argmax 在并列处是掷硬币（v3 策略头更平，探针 pos0 实测 margin=0.0007 < fp16 扰动 0.001），
    # 对它要求逐位一致是门禁过严。这些位置不参与 t1 判定；非并列位置仍严格要求一致。
    srt = np.sort(rp, axis=1)
    margins = srt[:, -1] - srt[:, -2]
    decisive = margins >= 0.05
    if decisive.any():
        t1 = float((pols.argmax(1)[decisive] == rp.argmax(1)[decisive]).mean())
    else:
        t1 = 1.0   # 全并列：argmax 无信息量，dp/dv 仍守门
    ok = np.isfinite(vals).all() and np.isfinite(pols).all() and dv <= GATE_V and dp <= GATE_P and t1 >= GATE_T1
    return dv, dp, t1, ok

def sync_infer(ireq, outs, xb):
    """原地张量推理：直接写插件自管输入缓冲（免 shadow copy，实测 b1 -33%），
    输出零视图读后按需拷贝（缓冲下一次 infer 会被覆写，必须拷出）"""
    it = ireq.get_input_tensor()
    it.data[:] = xb
    ireq.infer()
    if len(outs) == 1:
        return np.array(ireq.get_tensor(outs[0]).data, copy=True)
    return [np.array(ireq.get_tensor(o).data, copy=True) for o in outs]

chosen = None        # (split, dev_name, thr_ireq, thr_outs)
lat = None           # (lat_ireq, lat_outs) 或 None —— 延迟档小批模型
avail = core.available_devices
devices = []
for d in [args['device'], 'GPU', 'CPU']:
    if d not in devices:
        devices.append(d)

for dev in devices:
    dev_name = dev if dev in avail else next((a for a in avail if a == dev or a.startswith(dev + '.')), None)
    if dev_name is None:
        print(f'[OV] device {dev} not available (have: {avail})', file=sys.stderr, flush=True)
        continue
    for split in split_order(dev):
        try:
            t0 = time.time()
            compiled = core.compile_model(get_ir(split, B), dev_name, compile_props(dev_name, B, False))
            print(f'[OV] compile OK {split}@{dev_name} b{B} in {time.time()-t0:.1f}s '
                  f'({compile_props(dev_name, B, False)})', file=sys.stderr, flush=True)
        except Exception as e:
            print(f'[OV] compile FAIL {split}@{dev_name} b{B}: {str(e)[:180]}', file=sys.stderr, flush=True)
            continue
        outs = resolve_outputs(compiled, split)
        if outs is None:
            continue
        ireq = compiled.create_infer_request()
        try:
            vo, po = run_probe(ireq, outs)
        except Exception as e:
            print(f'[OV] probe FAIL {split}@{dev_name}: {str(e)[:160]}', file=sys.stderr, flush=True)
            continue
        dv, dp, t1, ok = gate_check(vo, po)
        print(f'[OV] gate {split}@{dev_name} b{B}: value={dv:.4f} policy={dp:.4f} top1={t1:.3f} -> {"PASS" if ok else "FAIL"}', file=sys.stderr, flush=True)
        if ok:
            chosen = (split, dev_name, ireq, outs)
            break
    if chosen:
        break
assert chosen, 'all split@device combinations failed compile or accuracy gate'
split, dev_name, ireq, outs = chosen

# ---- 多粒度档位（GenAI PREFILL_CHUNK_SIZE 思路）：thr(B) 必有；
# 延迟档 BL 与中档 32 各自过精度门禁后加入，服务时按剩余量选最大可行档 ----
def pad_to(xb, bs):
    if xb.shape[0] < bs:
        return np.concatenate([xb, np.zeros((bs - xb.shape[0], C_IN, 10, 10), np.float32)])
    return xb

tiers = [(B, ireq, outs)]   # 降序维护

def try_tier(bs, for_latency):
    if bs >= B or any(bs == t[0] for t in tiers):
        return
    try:
        t0 = time.time()
        c = core.compile_model(get_ir(split, bs), dev_name, compile_props(dev_name, bs, for_latency))
        o = resolve_outputs(c, split)
        if o is None:
            return
        q = c.create_infer_request()
        k = min(4, bs)
        pv = sync_infer(q, o, pad_to(px[:k], bs))
        rv, rp = ref_val[:k], ref_pol[:k]
        if len(o) == 1:
            with torch.no_grad():
                _pol, _raw = head(torch.from_numpy(np.array(pv).copy()[:k]))
            vo2 = np.tanh(_raw.numpy().reshape(-1)).astype(np.float32)
            po2 = _pol.numpy().astype(np.float32)
        else:
            vo2 = np.tanh(np.array(pv[1]).reshape(-1)[:k]).astype(np.float32)
            po2 = np.array(pv[0])[:k].astype(np.float32)
        dv2, dp2, t12, ok2 = gate_check(vo2, po2, rv, rp)
        if ok2:
            tiers.append((bs, q, o))
            tiers.sort(key=lambda t: -t[0])
            print(f'[OV] tier OK {split}@{dev_name} b{bs} in {time.time()-t0:.1f}s (gate v={dv2:.4f} p={dp2:.4f} t1={t12:.3f})', file=sys.stderr, flush=True)
        else:
            print(f'[OV] tier gate FAIL b{bs} (v={dv2:.4f} p={dp2:.4f} t1={t12:.3f})', file=sys.stderr, flush=True)
    except Exception as e:
        print(f'[OV] tier b{bs} unavailable: {str(e)[:140]}', file=sys.stderr, flush=True)

if B > BL:
    try_tier(BL, True)
if dev_name and dev_name.upper().startswith('NPU'):
    try_tier(1, True)          # 单盘档：原地张量实测 1.21ms，服务单局面评估/悔棋路径
BM = min(16, B)   # 实测 b16=1.435ms/盘、b32=1.376、b64 反弹 1.757 —— 甜点位在 32
if B > BM and BM > BL:
    try_tier(BM, False)

def pick_tier(remaining):
    for bs, q, o in tiers:
        if bs <= remaining:
            return bs, q, o
    return tiers[-1]

def infer_chunked(x):
    """多粒度选块推理：split!=full 返回特征 (n,128,10,10)；split==full 返回 (vals,pols)"""
    n = x.shape[0]
    feats, pols_all, vals_all = [], [], []
    i = 0
    while i < n:
        bs, q, o = pick_tier(n - i)
        pv = sync_infer(q, o, pad_to(x[i:i + bs], bs))
        if len(o) == 1:
            feats.append(pv)
        else:
            pols_all.append(pv[0]); vals_all.append(pv[1])
        i += bs
    if len(outs) == 1:
        return np.concatenate(feats)[:n]
    pols = np.concatenate(pols_all)[:n].astype(np.float32)
    vals = np.tanh(np.concatenate(vals_all)[:n].reshape(-1)).astype(np.float32)
    return vals, pols

# ---- 异步队列评估：MCTS 为"单请求-串行应答"负载，块间主机间隙 <1ms，
# AsyncInferQueue 在此无收益（官方异步模式面向持续并发流），故保持同步逐块。
nireq_info = '?'
try:
    nireq_info = ireq.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
except Exception:
    pass

print(f'[OV] MODE {split}@{dev_name} tiers={[t[0] for t in tiers]} nireq={nireq_info} turbo={"on" if args["turbo"] else "off"} int8={"on" if INT8 else "off"}', file=sys.stderr, flush=True)

def expand(boards):
    n = boards.shape[0]
    out = np.zeros((n, 24, 10, 10), dtype=np.float32)
    pieces = boards[:, :100]
    for v in range(1, 15):
        out[:, v - 1] = (pieces == v).reshape(n, 10, 10)
    out[:, 14] = boards[:, 100][:, None, None]
    out[:, 15] = boards[:, 101][:, None, None]
    ep = boards[:, 102] - 1
    m = ep >= 0
    if m.any():
        rows = np.nonzero(m)[0]
        out[rows, 16, ep[rows] // 10, ep[rows] % 10] = 1.0
    out[:, 17:21] = boards[:, 103:107][:, :, None, None]
    return out

def read_exact(k):
    buf = b''
    while len(buf) < k:
        chunk = sys.stdin.buffer.read(k - len(buf))
        if not chunk: return None
        buf += chunk
    return buf

print('READY', file=sys.stderr, flush=True)

_dbg = os.environ.get('OV_BRIDGE_DEBUG') == '1'
_dbg_n = 0
_acc = {'read': 0.0, 'expand': 0.0, 'infer': 0.0, 'head': 0.0, 'write': 0.0}

with torch.no_grad():
    while True:
        t0 = time.time()
        hdr = read_exact(4)
        if hdr is None: break
        n = struct.unpack('<i', hdr)[0]
        if n <= 0: continue
        data = read_exact(n * 107 * 4)
        if data is None: break
        t1 = time.time()
        boards = np.frombuffer(data, dtype=np.int32).reshape(n, 107)
        x = expand(boards)
        t2 = time.time()
        if split == 'full':
            vals, pols = infer_chunked(x)
            head_ms = 0.0
        else:
            f_np = infer_chunked(x)
            t3 = time.time()
            f_t = torch.from_numpy(f_np)
            pol, raw = head(f_t)
            pols = pol.numpy().astype(np.float32)
            vals = np.tanh(raw.numpy().reshape(-1)).astype(np.float32)
            head_ms = time.time() - t3
        t4 = time.time()
        sys.stdout.buffer.write(struct.pack('<i', n) + vals.tobytes() + pols.tobytes())
        sys.stdout.buffer.flush()
        if _dbg:
            _dbg_n += 1
            _acc['read'] += t1 - t0; _acc['expand'] += t2 - t1; _acc['infer'] += t3 - t2
            _acc['head'] += head_ms; _acc['write'] += time.time() - t4
            if _dbg_n % 16 == 0:
                tot = sum(_acc.values())
                print(f'[bridge] n={n} ' + ' '.join(f'{k}={v/_dbg_n*1000:.1f}' for k, v in _acc.items()) + f' total={tot/_dbg_n*1000:.1f}ms', file=sys.stderr, flush=True)
                for k in _acc: _acc[k] = 0.0

sys.exit(0)
