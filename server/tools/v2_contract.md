# chess10 v2 前向契约（所有实现的唯一事实来源）

参考实现（数学权威）：`ov_train/az_model.py`（Net/ResBlock/GRN/MANO 类）。
对拍数据：`server/tools/v2_ref/`（manifest.json + weights_v2_rand.bin + 激活 .f32）。
公差：max|Δ| ≤ 2e-2（fp32 累加序差异可容忍；显著超出即为实现错误）。

## 1. 权重 bin v2 布局

前缀（2,834,213 floats）与 v1 完全相同（W_KEYS 28 项 + Wr0..11/br0..11/bng/bnb/bnm/bnv 各12）。
**末尾追加**（顺序严格）：

| 名 | 数量 | JS 语义 |
|---|---|---|
| grn0..grn5 | 6×128 | 各 ResBlock 尾 GRN γ（per-channel） |
| WqM/WkM/WvM/WoM | 各 16384 | MANO Linear 权重，布局 (out,in) row-major：`x@W.T`，即 W[(o*128)+i] |
| Dw | 65536 | MANO 降采样 conv k2s2，PyTorch Conv2d 布局 (out,in,2,2)：Dw[(o*128+i)*4 + kh*2 + kw] |
| Db | 128 | 降采样 bias |
| Uw | 65536 | MANO 上采样 ConvTranspose2d k2s2，布局 (in,out,2,2)：Uw[(i*128+o)*4 + kh*2 + kw] |
| Ub | 128 | 上采样 bias |
| ln_mg / ln_mb | 各 128 | MANO 共享 LayerNorm γ/β |
| rpbT | 1444 = 4×361 | 末段标准注意力 2D 相对位置偏置：rpbT[h*361 + (dr+9)*19 + (dc+9)] |

## 2. 网络前向（行棋方视角输入 24×10×10）

```
f = ReLU(FrozenBN(conv0(x)))
f = ResBlock0(f); f = ResBlock1(f); f = ResBlock2(f)     # 每块尾接 GRN
f = MANO(f)
f = ResBlock3(f); f = ResBlock4(f); f = ResBlock5(f)
t = flatten(f) → tokens (100,128)，token p 对应格 (r=p/10|0, c=p%10)
（末段标准注意力，post-norm，带 rpb）
pol = polo(ReLU(polc(feat))).flatten  → POLICY_CH*N_POS logits（v3：160 × 100 = 16000）
raw = vall2(ReLU(vall1(ReLU(valc(feat)).flatten)))      → 标量（tanh 由消费端加）
```

### ResBlock（与 v1 相同 + 尾部 GRN）
h = ReLU(BN1(conv1(f))); h = ReLU(BN2(conv2(h))); g = ReLU(f + h)
GRN: g_out = g + gamma_c ⊙ g / (sqrt(Σ_spatial g²) + 1e-6)   （γ 初始 0 → 恒等）

### MANO（window=5, levels=3, 4头，各级共享全部权重）
```
sizes: L0=10×10 → L1=5×5 → L2=2×2（D: conv 2×2 stride2 + Db）
total = 0
h = 输入
for lv in 0..2:
  if lv>0: h = down(h)
  u = window_attn(h)        # 共享 LN→QKV→分窗 softmax 注意力→Wo
  for m in 0..lv-1: u = up(u, target=sizes[lv-1-m])   # ConvTranspose 2×2 s2 + Ub，逐级回到目标尺寸
  total += u
out = 输入 + total
```
window_attn 细节：
- t = tokens；t = LayerNorm_m(t)（γ=ln_mg, β=ln_mb）
- Q=x@WqM.T, K=x@WkM.T, V=x@WvM.T（各 128）
- **分窗**：仅当 H>5 且 W>5 且整除（即 L0 10×10）：gh=gw=2，窗 (a,b) 覆盖行 [a*5,(a+1)*5) 列 [b*5,(b+1)*5)；窗内独立 softmax 注意力；否则（L1/L2）单窗=全图注意力
- 4 头 × 32 维：scores = Q_h·K_hᵀ/√32 → softmax → ×V_h → 拼回 → Wo
- Wo **零初始化起步**（对拍权重文件里已随机化）
- 上采样：convT 输出 (in-1)*2+2，若目标尺寸更大则**底部/右侧补零**到目标（对应 PyTorch output_size 语义）
- L2(2×2)→L1(5×5)：convT 得 4×4，右/下补 1 行列零；L1(5×5)→L0(10×10)：5×2=10 精确无补

### 末段标准注意力 + rpb
scores[h,i,j] = Q_h[i]·K_h[j]/√32 + rpbT[h*361 + (row_i-row_j+9)*19 + (col_i-col_j+9)]
（token p: row=p/10|0, col=p%10；rpb 零初始化起步，对拍文件里已随机化）
其余：post-norm `h=ln1(Wo@o+t)`、FFN 128→256→ReLU→128、`h2=ln2(ff2+ h)`——与 v1 一字不差。

## 3. 对拍数据

- `weights_v2_rand.bin`：新模块已随机化（seed 20260901）——**JS 测试必须用这份**，旧权重下新模块全零测不出布局错误
- `inputs.f32`：32×24×100（通道主序，与 encodeBoard 一致）
- `feat_trunk.f32`：32×128×100 trunk 全输出（含 MANO+GRN）
- `mano_out.f32`：32×128×100（仅 MANO 模块输出；输入=前3块后特征，可由 feat 前段自建或直接对照模块）
- `pol_logits.f32` / `raw_value.f32`：最终双头输出

## 4. 验收
- 恒等：v1 bin（weights_ov.bin）载入 v2 前向 == v1 前向（逐位）
- 对拍：weights_v2_rand.bin 下 JS 前向 vs 参考 .f32，max|Δ| ≤ 2e-2
- 加载器：文件长度严格校验（v1 尺寸=v1 兼容 / v2 尺寸=全量 / 其他=抛错），禁止静默截断

## 5. GPU 实现（gpu.js，2026-09-01 上线）

`gpu.js` 为第三份实现（python 权威 / cnn.js CPU / gpu.js OpenCL）。v2 权重（`__v2`）时 evalBatch 走 v2 编排，v1 权重路径被门控逐位不变。

### kernel 清单（本次新增）
| kernel | 作用 | 备注 |
|---|---|---|
| `grn_inplace` | 各 ResBlock 尾 GRN（通道主序就地） | per-(n,ch) 线程算 ‖g‖ 后缩放 |
| `conv2s2` | MANO 降采样 k2s2（Dw 两级共享） | 参数化 Hin/Win/Ho/Wo |
| `convT2s2` | MANO 上采样 k2s2（Uw 共享） | scatter 逆映射；目标尺寸大于原生 (in-1)*2+2 时底/右无贡献（=bias 残留，与补零语义一致） |
| `transpose_ct` / `transpose_tc` | 变长通道↔token 转置 | TOK=100/25/4 复用 |
| `mano_attn` | 融合窗口注意力 | 每线程一个 (b,t,h)；窗 origin=(r/WIN)*WIN；scores[MANO_WIN²]；与 cnn.js manoAttn 同构 |
| `rpb_add` | 末段注意力分数广播加 rpbEff | 位于 matmulBH 之后、softmax 之前 |

### 关键等价点
- **rpbEff 预乘 √HD**：GPU softmax 在 `/√32` 缩放**前**加 rpb（cnn.js 在缩放后）。`max(√32·y)=√32·max(y)` ⇒ 含 max 归一化在内逐点等价。上传时完成，勿删。
- **MANO QKV/Wo 转置**：权重为 (out,in)（cnn.js `Q[o]=Σ t[i]·W[o,i]`），上传时 `trMN` 转置为 matmul kernel 的 (in,out)。Wq/Wk/Wv/Wo（v1 末段）不转置。
- **缓冲区接力**：blk0-2 输出落 b.u → `runMano(N, b.u)` 就地写回 b.u → blk≥3 的 src/dst 奇偶翻转（u→f→u→f），末态落 b.f 与 v1 一致；每块 addrelu 后 GRN 作用于 dst。
- `isReady()` = init 且 upload 均成功；同一权重对象重复上传跳过（worker 与 mcts.loadWeights 双入口）。

### 验收记录（RTX 5070 Laptop，tools/gpu_v2_parity_test.js → ALL PASS）
| 对拍 | policy | value(tanh) |
|---|---|---|
| GPU vs python（口径：Δ/全局最大幅度） | 5.24e-4 | 8.9e-4 abs |
| GPU vs JS 同 32 参考盘（逐元素 rel） | 6.95e-5 | 7.84e-6 abs |
| GPU vs JS 24 随机盘 fuzz | 1.56e-4 | 7.21e-6 abs |
| v1 回归（evalBatch 改造后） | 9.52e-5 | 6.17e-6 abs |

vs python 的 policy 用「全局幅度」口径：python(Torch fp32/TF32 累积序) 对 fp64 参考在 ±550 logits 上本就有 ~0.3 绝对差（JS↔python 直拍实测 0.312，既有基线），逐元素相对会误报。GPU 正确性的紧口径证据是 vs JS（fp64 参考实现）三组。

- 审计：code-reviewer 0 blocker；已修 P2：zeroBias 分配 ≥256（覆盖 K=128，防 batch 回退后越界读）、删除 mtmp/tKhT/tSh 死 buffer（省 ~290MB/worker）、isReady 收紧 + 重复上传跳过。
- E2E：tools/e2e_v2_smoke.js —— worker ready 消息 `arch:'v2'`、响应 stats.evals>0（GPU 批量评估真实参与）、着法经 Engine 验证合法（含骑士特权双跳 g1→g5）；完整服务器 ws 通路同过。

### GRN γ 训练语义（决策记录）
γ 零初始化、wd=0、1×LR（新模块参数组），**不冻结**（审计曾议冻结；决策=参与训练。γ 仅 6×128 参数，且 `g/(‖g‖+ε)` 自约束幅度，风险可忽略）。torch_ov_train.py 的 new_ids 排除逻辑防止 γ 重复入组。
