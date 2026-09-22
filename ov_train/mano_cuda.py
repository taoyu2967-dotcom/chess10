# -*- coding: utf-8 -*-
# 手写 CUDA 融合算子：MANO 窗口注意力（窗重排 + QK^T/scale + softmax + AV + 反窗）fwd+bwd
# 对应 az_model.MANO._window_attn 的 attention 内核段（LN/仿射/QKV/Wo/上下采样仍走 PyTorch）
# 用法: py -3 mano_cuda.py            # 自带对拍门禁 + 四档基准
#       训练侧启用: from mano_cuda import patch_mano; patch_mano()
import os, sys, time
import torch
from torch.utils.cpp_extension import load_inline

# ---- Windows 编译链：MSVC(cl) 与 CUDA_HOME ----
_MSVC = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64'
_CUDA = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
if os.path.isdir(_MSVC):
    os.environ['PATH'] = _MSVC + os.pathsep + os.environ.get('PATH', '')
if os.path.isdir(_CUDA):
    os.environ['CUDA_HOME'] = _CUDA

D_MODEL, HEADS, HD = 128, 4, 32
MAXK = 32

_cpp = r'''
#include <torch/extension.h>
#include <vector>

// 纯 C 接口：.cu 侧不碰 torch 头（规避 MSVC 14.44 + torch 头在 nvcc 宿主段的 std 歧义）
// 指针统一 void*，由 is_half 选择 fp16/fp32 实例，AMP 下免去 8 次 cast kernel
extern "C" {
void mano_fwd_launch(const void* q, const void* k, const void* v, void* o, void* p,
                     int B, int HW, int nw, int keys, int Wd, int w, bool windowed, int is_half);
void mano_bwd_launch(const void* q, const void* k, const void* v, const void* dout, const void* p,
                     void* dq, void* dk, void* dv,
                     int B, int HW, int nw, int keys, int Wd, int w, bool windowed, int is_half);
}

void mano_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                   torch::Tensor o, torch::Tensor p,
                   int64_t HW, int64_t nw, int64_t keys, int64_t Wd, int64_t w, bool windowed) {
    int is_half = (q.scalar_type() == at::kHalf) ? 1 : 0;
    mano_fwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), p.data_ptr(),
                    (int)q.size(0), (int)HW, (int)nw, (int)keys, (int)Wd, (int)w, windowed, is_half);
}

std::vector<torch::Tensor> mano_attn_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                         torch::Tensor dout, torch::Tensor p,
                                         int64_t HW, int64_t nw, int64_t keys, int64_t Wd, int64_t w, bool windowed) {
    // kernel 全量覆写 dq/dk/dv，empty 即可（原 zeros 的 3 次 fill 纯浪费）
    auto dq = torch::empty_like(q), dk = torch::empty_like(q), dv = torch::empty_like(q);
    int is_half = (q.scalar_type() == at::kHalf) ? 1 : 0;
    mano_bwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), dout.data_ptr(), p.data_ptr(),
                    dq.data_ptr(), dk.data_ptr(), dv.data_ptr(),
                    (int)q.size(0), (int)HW, (int)nw, (int)keys, (int)Wd, (int)w, windowed, is_half);
    return {dq, dk, dv};
}
'''

_cuda = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MAX_KEYS 32

// blockIdx.x 扁平 = b*(nw*4) + win*4 + head；块内 32 线程 = HD 维
// windowed: 单元格 (r,c) 属于窗 (r/w, c/w)，窗内索引 (r%w)*w + (c%w)
//
// v6 要点：
// 1) IO 模板化 fp16/fp32（kernel 内部一律 fp32 计算；AMP 下 half 直进 kernel，
//    省 8 次 cast kernel 与一半 DRAM 流量。__CUDA_NO_HALF_* 宏只禁运算符，
//    __half2float/__float2half 函数仍可用）。
// 2) 双角色 lane：lane j (tid<keys) 负责 key/query j（QK+softmax，k/v 行驻寄存器，
//    softmax 用 warp shuffle）；lane d（全 32 线程）负责维度 d（AV / dq / 输出写出）。
//    shared 一律保持 lane 在最快维的 s[j][tid] 形态——[tid][j] 形态按 j 迭代是
//    32-way bank 冲突（v4 bwd 的教训）；q/dout 行直接 global 广播读。
// 3) AV/dq 的跨 lane 求和用 __shfl_sync 广播 p/dS，不再过 shared：fwd 与 bwd
//    各只剩 1 次 __syncthreads（仅为首读 v_s/k_s 可见性），smem 减半，
//    occupancy fwd 由 smem 限制的 12 块/SM 提到 23 块/SM。

__device__ __forceinline__ float ldv(const float* p) { return *p; }
__device__ __forceinline__ float ldv(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void stv(float* p, float v) { *p = v; }
__device__ __forceinline__ void stv(__half* p, float v) { *p = __float2half(v); }

__device__ __forceinline__ int cell_index(int win, int qi, int Wd, int w, bool windowed) {
    if (!windowed) return qi;
    int gw = Wd / w;
    return ((win / gw) * w + qi / w) * Wd + (win % gw) * w + qi % w;
}

template <typename T>
__global__ void mano_fwd_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                const T* __restrict__ v, T* __restrict__ o,
                                T* __restrict__ p,
                                int HW, int nw, int keys, int Wd, int w, bool windowed) {
    int rest = blockIdx.x;
    int head = rest % 4; rest /= 4;
    int win = rest % nw;
    int b = rest / nw;

    __shared__ float v_s[MAX_KEYS][32];        // v[j][d]：AV 阶段 lane d 按 j 读，无冲突

    int tid = threadIdx.x;
    const float scale = rsqrtf(32.f);
    const int pbase = ((b * nw + win) * 4 + head) * keys * MAX_KEYS;

    for (int j = 0; j < keys; j++) {
        int g = (b * HW + cell_index(win, j, Wd, w, windowed)) * 4 + head;
        v_s[j][tid] = ldv(v + g * 32 + tid);
    }
    float k_r[32];                             // lane j 独占 key j 的 k 行（寄存器）
    #pragma unroll
    for (int d = 0; d < 32; d++) k_r[d] = 0.f;
    if (tid < keys) {
        const T* kg = k + ((b * HW + cell_index(win, tid, Wd, w, windowed)) * 4 + head) * 32;
        #pragma unroll
        for (int d = 0; d < 32; d++) k_r[d] = ldv(kg + d);
    }
    __syncthreads();                           // 唯一 barrier：v_s 对跨 lane 读可见

    // QK^T/scale + softmax + AV：lane j 算 p[qi][j]（q 行 L1 广播读），AV 靠 shuffle
    // 把 p[qi][j] 广播给 lane d，全程零 barrier
    for (int qi = 0; qi < keys; qi++) {
        const T* qg = q + ((b * HW + cell_index(win, qi, Wd, w, windowed)) * 4 + head) * 32;
        float dot = 0.f;
        #pragma unroll
        for (int d = 0; d < 32; d++) dot += ldv(qg + d) * k_r[d];
        dot *= scale;
        float my = (tid < keys) ? dot : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) my = fmaxf(my, __shfl_down_sync(0xffffffff, my, off));
        float mx = __shfl_sync(0xffffffff, my, 0);
        float e0 = (tid < keys) ? __expf(dot - mx) : 0.f;
        float e = e0;                          // e 被规约原地累加，pv 必须用规约前的副本
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) e += __shfl_down_sync(0xffffffff, e, off);
        float pv = e0 * (1.f / __shfl_sync(0xffffffff, e, 0));
        if (tid < keys) stv(p + pbase + qi * MAX_KEYS + tid, pv);
        float acc = 0.f;                       // o[qi][d] = Σ_j p[qi][j]·v[j][d]
        for (int j = 0; j < keys; j++) acc += __shfl_sync(0xffffffff, pv, j) * v_s[j][tid];
        stv(o + ((b * HW + cell_index(win, qi, Wd, w, windowed)) * 4 + head) * 32 + tid, acc);
    }
}

template <typename T>
__global__ void mano_bwd_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                const T* __restrict__ v, const T* __restrict__ dout,
                                const T* __restrict__ p,
                                T* __restrict__ dq, T* __restrict__ dk, T* __restrict__ dv,
                                int HW, int nw, int keys, int Wd, int w, bool windowed) {
    int rest = blockIdx.x;
    int head = rest % 4; rest /= 4;
    int win = rest % nw;
    int b = rest / nw;

    __shared__ float k_s[MAX_KEYS][32];        // k_s[j][d]：dq 阶段 lane d 按 j 读，无冲突

    int tid = threadIdx.x;
    const float scale = rsqrtf(32.f);
    const int pbase = ((b * nw + win) * 4 + head) * keys * MAX_KEYS;

    float v_r[32], dv_r[32], dk_r[32];         // lane j 独占 key j 的 v 行与 dV/dK 累加器
    #pragma unroll
    for (int d = 0; d < 32; d++) { v_r[d] = 0.f; dv_r[d] = 0.f; dk_r[d] = 0.f; }
    for (int j = 0; j < keys; j++) {           // k 协同装载：lane tid 取第 tid 列，无冲突
        int g = (b * HW + cell_index(win, j, Wd, w, windowed)) * 4 + head;
        k_s[j][tid] = ldv(k + g * 32 + tid);
    }
    if (tid < keys) {
        const T* vg = v + ((b * HW + cell_index(win, tid, Wd, w, windowed)) * 4 + head) * 32;
        #pragma unroll
        for (int d = 0; d < 32; d++) v_r[d] = ldv(vg + d);
    }
    __syncthreads();                           // 唯一 barrier：k_s 对跨 lane 读可见

    // 逐 qi：dS[j]（lane j，warp shuffle 规约）+ dv/dk 寄存器累计 + dq 经 shuffle 广播 dS
    for (int qi = 0; qi < keys; qi++) {
        int gq = (b * HW + cell_index(win, qi, Wd, w, windowed)) * 4 + head;
        const T* dog = dout + gq * 32;
        const T* qg = q + gq * 32;
        float pv = (tid < keys) ? ldv(p + pbase + qi * MAX_KEYS + tid) : 0.f;
        float dp = 0.f;                        // dp_full[j] = dO[qi]·v[j]
        #pragma unroll
        for (int d = 0; d < 32; d++) dp += ldv(dog + d) * v_r[d];
        float pdp = (tid < keys) ? pv * dp : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) pdp += __shfl_down_sync(0xffffffff, pdp, off);
        float ds = pv * (dp - __shfl_sync(0xffffffff, pdp, 0));   // dS[j]=p[j]·(dp−Σp·dp)
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            dv_r[d] += pv * ldv(dog + d);      // dV[j][d] += p[qi][j]·dO[qi][d]
            dk_r[d] += (scale * ds) * ldv(qg + d); // dK[j][d] += scale·dS[qi][j]·q[qi][d]
        }
        float acc = 0.f;                       // dq[qi][d] = scale·Σ_j dS[qi][j]·k[j][d]
        for (int j = 0; j < keys; j++) acc += __shfl_sync(0xffffffff, ds, j) * k_s[j][tid];
        stv(dq + gq * 32 + tid, acc * scale);
    }

    if (tid < keys) {
        int g = ((b * HW + cell_index(win, tid, Wd, w, windowed)) * 4 + head) * 32;
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            stv(dv + g + d, dv_r[d]);
            stv(dk + g + d, dk_r[d]);
        }
    }
}

extern "C" {

void mano_fwd_launch(const void* q, const void* k, const void* v, void* o, void* p,
                     int B, int HW, int nw, int keys, int Wd, int w, bool windowed, int is_half) {
    dim3 g(B * nw * 4);
    if (is_half)
        mano_fwd_kernel<__half><<<g, 32>>>((const __half*)q, (const __half*)k, (const __half*)v,
                                           (__half*)o, (__half*)p, HW, nw, keys, Wd, w, windowed);
    else
        mano_fwd_kernel<float><<<g, 32>>>((const float*)q, (const float*)k, (const float*)v,
                                          (float*)o, (float*)p, HW, nw, keys, Wd, w, windowed);
}

void mano_bwd_launch(const void* q, const void* k, const void* v, const void* dout, const void* p,
                     void* dq, void* dk, void* dv,
                     int B, int HW, int nw, int keys, int Wd, int w, bool windowed, int is_half) {
    dim3 g(B * nw * 4);
    if (is_half)
        mano_bwd_kernel<__half><<<g, 32>>>((const __half*)q, (const __half*)k, (const __half*)v,
                                           (const __half*)dout, (const __half*)p,
                                           (__half*)dq, (__half*)dk, (__half*)dv,
                                           HW, nw, keys, Wd, w, windowed);
    else
        mano_bwd_kernel<float><<<g, 32>>>((const float*)q, (const float*)k, (const float*)v,
                                          (const float*)dout, (const float*)p,
                                          (float*)dq, (float*)dk, (float*)dv,
                                          HW, nw, keys, Wd, w, windowed);
}

}
'''

_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name='chess10_mano_attn_v6',
            cpp_sources=_cpp, cuda_sources=_cuda,
            functions=['mano_attn_fwd', 'mano_attn_bwd'],
            verbose=False)
    return _mod


class ManoAttn(torch.autograd.Function):
    """融合窗口注意力。q,k,v: (B, HW, 128) 连续 fp32；返回 (B, HW, 128)。"""

    @staticmethod
    def forward(ctx, q, k, v, H, Wd, w, windowed):
        mod = _get_mod()
        B, HW = q.shape[0], q.shape[1]
        keys = (w * w) if windowed else HW
        nw = (H // w) * (Wd // w) if windowed else 1
        o = torch.empty_like(q)
        p = torch.empty(B * nw * HEADS * keys * MAXK, device=q.device, dtype=q.dtype)
        mod.mano_attn_fwd(q, k, v, o, p, HW, nw, keys, Wd, w, windowed)
        ctx.save_for_backward(q, k, v, p)
        ctx.cfg = (H, Wd, w, windowed, keys, nw)
        return o

    @staticmethod
    def backward(ctx, go):
        q, k, v, p = ctx.saved_tensors
        H, Wd, w, windowed, keys, nw = ctx.cfg
        mod = _get_mod()
        # v5：fp16 直进 kernel（内部 fp32 计算），dq/dk/dv 与输入同 dtype，无任何 cast
        dq, dk, dv = mod.mano_attn_bwd(q, k, v, go.contiguous(), p,
                                       q.shape[1], nw, keys, Wd, w, windowed)
        return dq, dk, dv, None, None, None, None


def fused_window_attention(q, k, v, H, Wd, w=5, windowed=True):
    """q,k,v: (B, HW, 128) 连续 fp32 → o: (B, HW, 128)（等价窗重排+MHA+反窗）"""
    return ManoAttn.apply(q.contiguous(), k.contiguous(), v.contiguous(), H, Wd, w, windowed)


# ================= eager 参照与自验 =================

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


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    dev = torch.device('cuda')
    _get_mod()
    print('=== 扩展编译完成，开始对拍 ===')

    cases = [(10, 10, 5, True), (5, 5, 5, False), (2, 2, 2, False)]
    B = 256
    all_ok = True
    for H, Wd, w, win in cases:
        HW = H * Wd
        q = torch.randn(B, HW, D_MODEL, device=dev)
        k = torch.randn(B, HW, D_MODEL, device=dev)
        v = torch.randn(B, HW, D_MODEL, device=dev)
        o_ref = _eager_window_attn(q, k, v, H, Wd, w, win)          # (B,C,H,Wd)
        o_new = fused_window_attention(q, k, v, H, Wd, w, win)      # (B,HW,C)
        o_ref_seq = o_ref.flatten(2).transpose(1, 2)                # (B,HW,C) 同布局
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
        fused_window_attention(q2, k2, v2, H, Wd, w, win).backward(g)
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

    print('=== attention 内核段基准 (B=256) ===')
    H, Wd, w, win = 10, 10, 5, True
    HW = H * Wd
    q = torch.randn(B, HW, D_MODEL, device=dev)
    k = torch.randn(B, HW, D_MODEL, device=dev)
    v = torch.randn(B, HW, D_MODEL, device=dev)
    t_eager = bench(lambda: _eager_window_attn(q, k, v, H, Wd, w, win))
    t_fused = bench(lambda: fused_window_attention(q, k, v, H, Wd, w, win))
    print(f'level0 窗口注意力: eager={t_eager:.3f}ms  fused={t_fused:.3f}ms  ({t_eager / t_fused:.2f}x)')

    # ---- 整网四档基准 ----
    import az_model as A
    import torch.nn.functional as F
    net = A.build_net(A._repo_root() + '/weights/BJ1_r208_v3.bin').to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True)
    x = torch.randn(B, A.C_IN, 10, 10, device=dev)
    pi = torch.zeros(B, A.PCH * A.N_POS, device=dev); pi[:, 0] = 1.0
    z = torch.zeros(B, device=dev)

    def timed_step(use_fused, use_amp):
        if use_fused:
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
    for name, uf, ua in [('eager fp32', False, False), ('eager + AMP', False, True),
                         ('fused CUDA fp32', True, False), ('fused CUDA + AMP', True, True)]:
        try:
            t = timed_step(uf, ua)
            print(f'  {name:18s} {t:6.1f} ms/步')
        except Exception as e:
            print(f'  {name:18s} 失败: {str(e)[:120]}')


_patched = False
_orig = None


def patch_mano():
    """把 MANO._window_attn 的 attention 段换成融合核（整网启用）。"""
    global _patched, _orig
    if _patched:
        return
    from az_model import MANO
    _orig = MANO._window_attn

    def _fused(self, x, lv):
        B, C, H, Wd = x.shape
        w = 5
        t = x.flatten(2).transpose(1, 2)
        t = self.ln(t)
        t = t * self.plg[lv] + self.plb[lv]
        q, k, v = self.Wq(t), self.Wk(t), self.Wv(t)
        windowed = self.window_on and H > w and Wd > w and H % w == 0 and Wd % w == 0
        o = fused_window_attention(q, k, v, H, Wd, w, windowed)     # (B, HW, C)，已是全局行主序
        o = self.Wo(o)
        o = o.transpose(1, 2).reshape(B, C, H, Wd)                  # seq→图像布局（窗序由 kernel 内部处理）
        return o

    MANO._window_attn = _fused
    _patched = True


def unpatch_mano():
    global _patched
    if _patched:
        from az_model import MANO
        MANO._window_attn = _orig
        _patched = False


if __name__ == '__main__':
    main()
