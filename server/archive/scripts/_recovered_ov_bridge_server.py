# -*- coding: utf-8 -*-
# OpenVINO 混合推理桥：NPU 主干(OV IR) + CPU fp32 注意力/双头(torch)
# 协议(stdin/stdout 二进制): REQ=[int32 N][N*107 int32] → RESP=[int32 N][N f32 values][N*10000 f32 policies]
# 就绪信号: stderr 输出 READY
import os, struct, sys, time
import numpy as np
import torch
import openvino as ov

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from az_model import build_net, Head, C_IN

def parse_args(argv):
    a = {'weights': None, 'ir': None, 'device': 'NPU', 'batch': 32, 'head_device': 'cpu', 'head_fp32': '1'}
    i = 0
    while i < len(argv):
        if argv[i] == '--weights': a['weights'] = argv[i + 1]
        elif argv[i] == '--ir': a['ir'] = argv[i + 1]
        elif argv[i] == '--device': a['device'] = argv[i + 1]
        elif argv[i] == '--batch': a['batch'] = int(argv[i + 1])
        elif argv[i] == '--head-device': a['head_device'] = argv[i + 1]
        elif argv[i] == '--head-fp32': a['head_fp32'] = argv[i + 1]
        i += 1
    return a

args = parse_args(sys.argv[1:])
B = args['batch']

net = build_net(args['weights'])
head = Head(net).eval()

core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cache')})
if args['ir'] and os.path.exists(args['ir']):
    ov_trunk = core.read_model(args['ir'])
else:
    from az_model import TrunkOnly
    ov_trunk = ov.convert_model(TrunkOnly(net), example_input=torch.zeros(B, C_IN, 10, 10), input=(B, C_IN, 10, 10))
compiled = core.compile_model(ov_trunk, args['device'])
ireq = compiled.create_infer_request()

# 头部子网络：默认 torch CPU fp32；--head-device <OV设备> 时编译到该设备（GPU 建议 fp32 保证精度）
head_req = None
head_dev = args['head_device']
if head_dev != 'cpu':
    from az_model import C_HID
    head_model = ov.convert_model(head, example_input=torch.zeros(B, C_HID, 10, 10), input=(B, C_HID, 10, 10))
    props = {}
    if args['head_fp32'] == '1':
        props[ov.properties.hint.inference_precision] = ov.Type.f32
    head_compiled = core.compile_model(head_model, head_dev, props)
    head_req = head_compiled.create_infer_request()

def expand(boards):
    """(N,107) int32 → (N,24,10,10) float32，与 cnn.encodeBoard 语义一致"""
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
_dbg_acc = {'expand': 0.0, 'trunk': 0.0, 'head': 0.0, 'write': 0.0, 'read': 0.0}

with torch.no_grad():
    while True:
        _t0 = time.time()
        hdr = read_exact(4)
        if hdr is None: break
        n = struct.unpack('<i', hdr)[0]
        if n <= 0: continue
        data = read_exact(n * 107 * 4)
        if data is None: break
        _t1 = time.time()
        boards = np.frombuffer(data, dtype=np.int32).reshape(n, 107)
        x = expand(boards)
        _t2 = time.time()
        feats = []
        for i in range(0, n, B):
            xb = x[i:i + B]
            if xb.shape[0] < B:
                xb = np.concatenate([xb, np.zeros((B - xb.shape[0], C_IN, 10, 10), np.float32)])
            ireq.infer({0: xb})
            feats.append(ireq.get_output_tensor(0).data[:].copy())
        f_np = np.concatenate(feats)[:n]
        _t3 = time.time()
        if head_req is not None:
            # 头部子网络在 OV 设备上（iGPU fp32：精度已验证 0 误差），按 B 分块
            pols_list, vals_list = [], []
            for i in range(0, n, B):
                fb = f_np[i:i + B]
                nb_ = fb.shape[0]
                if nb_ < B:
                    fb = np.concatenate([fb, np.zeros((B - nb_, f_np.shape[1], 10, 10), np.float32)])
                head_req.infer({0: fb})
                pols_list.append(head_req.get_output_tensor(0).data[:nb_].copy())
                vals_list.append(head_req.get_output_tensor(1).data[:nb_].copy())
            pols = np.ascontiguousarray(np.concatenate(pols_list)).astype(np.float32)
            vals = np.tanh(np.concatenate(vals_list)).astype(np.float32)
        else:
            f = torch.from_numpy(f_np)
            pol, raw = head(f)
            vals = np.tanh(raw.numpy()).astype(np.float32)
            pols = pol.numpy().astype(np.float32)
        _t4 = time.time()
        sys.stdout.buffer.write(struct.pack('<i', n) + vals.tobytes() + pols.tobytes())
        sys.stdout.buffer.flush()
        _t5 = time.time()
        if _dbg:
            _dbg_n += 1
            _dbg_acc['read'] += _t1 - _t0; _dbg_acc['expand'] += _t2 - _t1
            _dbg_acc['trunk'] += _t3 - _t2; _dbg_acc['head'] += _t4 - _t3
            _dbg_acc['write'] += _t5 - _t4
            if _dbg_n % 8 == 0:
                tot = sum(_dbg_acc.values())
                print(f'[bridge] n={n} read={_dbg_acc["read"]/_dbg_n*1000:.1f} expand={_dbg_acc["expand"]/_dbg_n*1000:.1f} trunk={_dbg_acc["trunk"]/_dbg_n*1000:.1f} head={_dbg_acc["head"]/_dbg_n*1000:.1f} write={_dbg_acc["write"]/_dbg_n*1000:.1f} total={tot/_dbg_n*1000:.1f}ms', file=sys.stderr, flush=True)
                for k in _dbg_acc: _dbg_acc[k] = 0.0
sys.exit(0)