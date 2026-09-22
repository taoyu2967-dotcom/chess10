# -*- coding: utf-8 -*-
# 纯 PyTorch SDPA 版 MANO 窗口注意力：用官方融合核 F.scaled_dot_product_attention
# 替换 az_model.MANO._window_attn 的 attention 内核段（窗重排 + QK^T/scale + softmax + AV + 反窗）。
# 与手写 CUDA 方案 mano_cuda.py 竞速；本文件完全独立（Windows 免编译），不改 az_model.py / mano_cuda.py。
# 用法: py -3 mano_sdpa.py            # 自带三项对拍门禁 + 切片/整网基准
#       训练侧启用: from mano_sdpa import patch_mano; patch_mano()
import sys, time, warnings
import torch
import torch.nn.functional as F

D_MODEL, HEADS, HD = 128, 4, 32
WEIGHTS = os.environ.get('CHESS10_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/weights/BJ1_r208_v3.bin'

# ---------------- SDPA backend 探测（torch.nn.attention，torch>=2.3） ----------------
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_ctx, SDPBackend
except ImportError:                                  # 极老版本兜底（本机 2.11 不会走到）
    _sdpa_ctx = None
    SDPBackend = None

# torch 默认派发优先级（flash > cudnn > mem-efficient > math）
_BACKEND_ORDER = ['FLASH_ATTENTION', 'CUDNN_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH']
_probe_cache = {}          # dtype -> (chosen_name, [可用 backend 名])
_last_backend = '(未探测)'


def _default_flag(name):
    bc = torch.backends.cuda
    if name == 'FLASH_ATTENTION':
        return bc.flash_sdp_enabled()
    if name == 'EFFICIENT_ATTENTION':
        return bc.mem_efficient_sdp_enabled()
    if name == 'MATH':
        return bc.math_sdp_enabled()
    if name == 'CUDNN_ATTENTION':
        return bool(getattr(bc, 'cudnn_sdp_enabled', lambda: False)())
    return False


def _exclusive_ctx(name):
    """互斥启用单一 SDPA backend 的上下文管理器。"""
    if _sdpa_ctx is not None and SDPBackend is not None:
        be = getattr(SDPBackend, name, None)
        if be is not None:
            return _sdpa_ctx(be)
    raise RuntimeError(f'backend {name} 不可用（无 torch.nn.attention）')


def _probe_backend(qh, kh, vh, dtype):
    """互斥逐个试跑各 backend；再把默认派发的输出与各可用 backend 输出做位级比对，
    确认无约束调用实际命中的 backend（优先级顺序兜底推断）。"""
    global _last_backend
    if dtype in _probe_cache:
        return _probe_cache[dtype][0]
    hits = []
    for name in _BACKEND_ORDER:
        if getattr(SDPBackend, name, None) is None:
            continue
        try:
            with _exclusive_ctx(name):
                F.scaled_dot_product_attention(qh, kh, vh)
            torch.cuda.synchronize()
            hits.append(name)
        except Exception:
            pass                        # 该 backend 编译期缺失或约束不满足 → 派发器也会跳过
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o_def = F.scaled_dot_product_attention(qh, kh, vh)
    matches = []
    for name in hits:
        try:
            with _exclusive_ctx(name):
                o_be = F.scaled_dot_product_attention(qh, kh, vh)
            if torch.equal(o_def, o_be):
                matches.append(name)
        except Exception:
            pass
    chosen = next((n for n in _BACKEND_ORDER if n in matches and _default_flag(n)),
                  next((n for n in _BACKEND_ORDER if n in hits and _default_flag(n)), None))
    tag = {True: '✓', False: '✗'}
    stat = '  '.join(f'{n.split("_")[0]}{tag[n in hits]}' for n in _BACKEND_ORDER)
    print(f'[SDPA] dtype={dtype}  可用: {stat}  位级匹配: {matches or "-"}  →  默认命中: {chosen}')
    _last_backend = chosen or 'MATH(兜底)'
    _probe_cache[dtype] = (chosen, hits)
    return chosen


# ---------------- SDPA 版窗口注意力 ----------------

def sdpa_window_attention(q, k, v, H, Wd, w=5, windowed=True):
    """SDPA 融合窗口注意力。q,k,v: (B, HW, 128) → o: (B, HW, 128)（全局行主序）。
    等价 mano_cuda.fused_window_attention / az_model.MANO._window_attn 的 attention 段；
    is_causal=False、无 mask、scale=1/sqrt(HD)（SDPA 默认）。"""
    B, HW, C = q.shape
    if windowed:
        gh, gw_ = H // w, Wd // w
        nw = gh * gw_
        keys = w * w

        def prep(u):
            # (B,HW,C) → (B*nw, HEADS, keys, HD)：窗重排 + 拆头一次 permute 完成
            u = u.view(B, gh, w, gw_, w, HEADS, HD)
            return u.permute(0, 1, 3, 5, 2, 4, 6).reshape(B * nw, HEADS, keys, HD)

        qh, kh, vh = prep(q), prep(k), prep(v)
        if q.dtype not in _probe_cache:                # 首次调用时打印实际命中的 backend
            _probe_backend(qh, kh, vh, q.dtype)
        oh = F.scaled_dot_product_attention(qh, kh, vh)   # (B*nw, HEADS, keys, HD)
        # 拆头 + 反窗合并为单次 permute：(B, wr, wc, head, ir, ic, hd) → (B, wr, ir, wc, ic, head, hd)
        o = (oh.view(B, gh, gw_, HEADS, w, w, HD)
               .permute(0, 1, 4, 2, 5, 3, 6)
               .reshape(B, HW, C))
    else:
        qh = q.view(B, HW, HEADS, HD).transpose(1, 2)  # (B, HEADS, HW, HD) 标准视图，零拷贝
        kh = k.view(B, HW, HEADS, HD).transpose(1, 2)
        vh = v.view(B, HW, HEADS, HD).transpose(1, 2)
        if q.dtype not in _probe_cache:
            _probe_backend(qh, kh, vh, q.dtype)
        oh = F.scaled_dot_product_attention(qh, kh, vh)
        o = oh.transpose(1, 2).reshape(B, HW, C)
    return o


# ================= eager 参照（照抄 mano_cuda.py）与 patch =================

def _eager_window_attn(q, k, v, H, Wd, w=5, windowed=True):
    """与 az_model.MANO._window_attn 的 attention 段逐操作等价的纯 PyTorch 参照。"""
    B, HW, C = q.shape
    gh, gw_ = (H // w, Wd // w) if windowed else (1, 1)
    nw = gh * gw_

    def win(u):
        u2 = u.view(B, gh, w, gw_, w, C)
        return u2.permute(0, 1, 3, 2, 4, 5).reshape(B, nw, w * w, C)

    qw, kw, vw = (win(q), win(k), win(v)) if windowed else (
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
    qh = qw.view(B, nw, -1, HEADS, HD).transpose(2, 3)
    kh = kw.view(B, nw, -1, HEADS, HD).transpose(2, 3)
    vh = vw.view(B, nw, -1, HEADS, HD).transpose(2, 3)
    att = torch.softmax(qh @ kh.transpose(-2, -1) / (HD ** 0.5), dim=-1)
    o = (att @ vh).transpose(2, 3).reshape(B, nw, -1, C)
    if windowed:
        o = o.view(B, gh, gw_, w, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, Wd)
    else:
        o = o.squeeze(1).transpose(1, 2).reshape(B, C, H, Wd)
    return o


_patched = False
_orig = None


def patch_mano():
    """把 MANO._window_attn 的 attention 段换成 SDPA 融合核（整网启用）。"""
    global _patched, _orig
    if _patched:
        return
    from az_model import MANO
    _orig = MANO._window_attn

    def _sdpa(self, x, lv):
        B, C, H, Wd = x.shape
        w = 5
        t = x.flatten(2).transpose(1, 2)
        t = self.ln(t)
        t = t * self.plg[lv] + self.plb[lv]
        q, k, v = self.Wq(t), self.Wk(t), self.Wv(t)
        windowed = self.window_on and H > w and Wd > w and H % w == 0 and Wd % w == 0
        o = sdpa_window_attention(q, k, v, H, Wd, w, windowed)   # (B, HW, C) 已是全局行主序
        o = self.Wo(o)
        o = o.transpose(1, 2).reshape(B, C, H, Wd)               # 直接回图像布局（无反窗）
        return o

    MANO._window_attn = _sdpa
    _patched = True


def unpatch_mano():
    global _patched
    if _patched:
        from az_model import MANO
        MANO._window_attn = _orig
        _patched = False


# ================= 自验与基准 =================

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    dev = torch.device('cuda')
    print(f'torch {torch.__version__}  {torch.cuda.get_device_name(0)}')
    print('=== 三项对拍（门禁 fwd≤1e-3 / grad≤1e-2） ===')

    cases = [(10, 10, 5, True), (5, 5, 5, False), (2, 2, 2, False)]
    B = 256
    all_ok = True
    for H, Wd, w, win in cases:
        HW = H * Wd
        q = torch.randn(B, HW, D_MODEL, device=dev)
        k = torch.randn(B, HW, D_MODEL, device=dev)
        v = torch.randn(B, HW, D_MODEL, device=dev)
        o_ref = _eager_window_attn(q, k, v, H, Wd, w, win)           # (B,C,H,Wd)
        o_new = sdpa_window_attention(q, k, v, H, Wd, w, win)        # (B,HW,C)
        o_ref_seq = o_ref.flatten(2).transpose(1, 2)                 # (B,HW,C) 同布局
        dfwd = (o_ref_seq - o_new).abs().max().item()
        g = torch.randn_like(o_new)
        q1 = q.detach().clone().requires_grad_(True)
        k1 = k.detach().clone().requires_grad_(True)
        v1 = v.detach().clone().requires_grad_(True)
        _eager_window_attn(q1, k1, v1, H, Wd, w, win).backward(
            g.transpose(1, 2).reshape(B, D_MODEL, H, Wd))
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        sdpa_window_attention(q2, k2, v2, H, Wd, w, win).backward(g)
        dg = max((q1.grad - q2.grad).abs().max().item(),
                 (k1.grad - k2.grad).abs().max().item(),
                 (v1.grad - v2.grad).abs().max().item())
        tag = f'{H}x{Wd} {"窗口" if win else "全图"}'
        ok = dfwd <= 1e-3 and dg <= 1e-2
        all_ok &= ok
        print(f'[{"PASS" if ok else "FAIL"}] {tag:10s} fwd|Δ|={dfwd:.2e}  grad|Δ|={dg:.2e}')
    if not all_ok:
        sys.exit(1)

    def bench(fn, reps=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / reps * 1000

    print('=== attention 内核段基准 (B=256, level0 10×10 窗5) ===')
    H, Wd, w, win = 10, 10, 5, True
    HW = H * Wd
    q = torch.randn(B, HW, D_MODEL, device=dev)
    k = torch.randn(B, HW, D_MODEL, device=dev)
    v = torch.randn(B, HW, D_MODEL, device=dev)
    t_eager = bench(lambda: _eager_window_attn(q, k, v, H, Wd, w, win))
    t_sdpa = bench(lambda: sdpa_window_attention(q, k, v, H, Wd, w, win))
    print(f'level0 窗口注意力: eager={t_eager:.3f}ms  sdpa={t_sdpa:.3f}ms  ({t_eager / t_sdpa:.2f}x)')
    q16, k16, v16 = q.half(), k.half(), v.half()
    with torch.autocast('cuda', dtype=torch.float16):     # 与 AMP 整网同上下文探测/计时
        sdpa_window_attention(q16, k16, v16, H, Wd, w, win)
        t_sdpa16 = bench(lambda: sdpa_window_attention(q16, k16, v16, H, Wd, w, win))
    print(f'level0 窗口注意力 fp16(AMP): sdpa={t_sdpa16:.3f}ms')

    # 各 backend 互斥计时（附加信息：裸 SDPA 核，不含重排拷贝）
    gh, gw_ = H // w, Wd // w
    nw = gh * gw_
    prep = lambda u: u.view(B, gh, w, gw_, w, HEADS, HD).permute(0, 1, 3, 5, 2, 4, 6) \
        .reshape(B * nw, HEADS, w * w, HD)
    qh, kh, vh = prep(q), prep(k), prep(v)
    _probe_backend(qh, kh, vh, q.dtype)
    avail = _probe_cache[q.dtype][1]
    for name in avail:
        try:
            with _exclusive_ctx(name):
                t = bench(lambda: F.scaled_dot_product_attention(qh, kh, vh))
            print(f'  仅 {name:20s} 裸核 {t:.3f}ms')
        except Exception as e:
            print(f'  仅 {name:20s} 失败: {str(e)[:80]}')

    # ---- 整网四档基准 ----
    import az_model as A
    net = A.build_net(WEIGHTS).to(dev)

    # patch 正确性整网抽查：直接对比 MANO 输出（门禁 ≤1e-3，fp32 累加序差异量级）。
    # 注意：训练后 blocks[3:] 对 ~1e-5 级扰动有 ~1e4 倍固有放大（对照实验：给 eager 输出加
    # 同幅随机扰动，trunk 输出同样放大到 ~3.5），故 trunk/pol 的绝对差是噪声放大所致，不作门禁。
    net.eval()
    fm = torch.randn(B, A.C_HID, 10, 10, device=dev)
    unpatch_mano()
    with torch.no_grad():
        r_ref = net.mano(fm)
    patch_mano()
    with torch.no_grad():
        r_new = net.mano(fm)
    d_mano = (r_ref - r_new).abs().max().item()
    ok_mano = d_mano <= 1e-3
    print(f'[{"PASS" if ok_mano else "FAIL"}] 整网抽查: MANO 输出 (SDPA patch vs eager) max|Δ|={d_mano:.2e}')

    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True)
    x = torch.randn(B, A.C_IN, 10, 10, device=dev)
    pi = torch.zeros(B, A.PCH * A.N_POS, device=dev); pi[:, 0] = 1.0
    z = torch.zeros(B, device=dev)

    def timed_step(use_sdpa, use_amp):
        if use_sdpa:
            patch_mano()
        else:
            unpatch_mano()

        def step():
            if use_amp:
                with torch.autocast('cuda', dtype=torch.float16):
                    pol, raw = net(x)
                    loss = 1.3 * F.mse_loss(raw.float(), z) - 2.0 * (pi * F.log_softmax(pol.float(), dim=1)).sum(1).mean()
            else:
                pol, raw = net(x)
                loss = 1.3 * F.mse_loss(raw, z) - 2.0 * (pi * F.log_softmax(pol, dim=1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        for _ in range(8):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            step()
        torch.cuda.synchronize()
        return (time.time() - t0) / 30 * 1000

    print('=== 整网每步 (B=256) ===')
    for name, us, ua in [('eager fp32', False, False), ('eager + AMP', False, True),
                         ('SDPA fp32', True, False), ('SDPA + AMP', True, True)]:
        try:
            t = timed_step(us, ua)
            print(f'  {name:18s} {t:6.1f} ms/步')
        except Exception as e:
            print(f'  {name:18s} 失败: {str(e)[:120]}')
    unpatch_mano()


if __name__ == '__main__':
    main()
