// int[107] → 24 通道 onehot（通道主序 (C, 100)）
__kernel void encode(__global const int* b, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * C_IN * 100;
  if (gid >= total) return;
  int n = gid / (C_IN * 100);
  int rem = gid % (C_IN * 100);
  int ch = rem / 100;
  int pos = rem % 100;
  int v = b[n * 107 + pos];   // 0=空, 1..14 = type*2+color+1（0-6 己方, 7-13 敌方）
  float val = 0.0f;
  if (ch < 14) {
    val = (ch == v - 1) ? 1.0f : 0.0f;
  } else if (ch == 14) {
    val = (float)b[n * 107 + 100];
  } else if (ch == 15) {
    val = (float)b[n * 107 + 101];
  } else if (ch == 16) {
    val = (b[n * 107 + 102] == pos + 1) ? 1.0f : 0.0f;
  } else if (ch >= 17 && ch <= 20) {
    val = (float)b[n * 107 + (103 + (ch - 17))];
  } else {
    val = 0.0f;
  }
  out[gid] = val;
}

__kernel void conv3x3(__global const float* in, __global const float* w,
                      __global const float* b, __global float* out,
                      int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  int r = pos / 10, c = pos % 10;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) {
    for (int kr = 0; kr < 3; kr++) {
      int rr = r + kr - 1;
      if (rr < 0 || rr >= 10) continue;
      for (int kc = 0; kc < 3; kc++) {
        int cc = c + kc - 1;
        if (cc < 0 || cc >= 10) continue;
        sum += in[(n*Cin + ic)*100 + rr*10 + cc] * w[((oc*Cin + ic)*3 + kr)*3 + kc];
      }
    }
  }
  out[(n*Cout + oc)*100 + pos] = sum;
}

__kernel void conv1x1(__global const float* in, __global const float* w,
                      __global const float* b, __global float* out,
                      int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) sum += in[(n*Cin + ic)*100 + pos] * w[oc*Cin + ic];
  out[(n*Cout + oc)*100 + pos] = sum;
}

// BN(scale+shift) + ReLU（通道主序）
__kernel void bnrelu(__global const float* in, __global const float* g,
                     __global const float* beta, __global const float* mean,
                     __global const float* var, __global float* out,
                     int N, int C) {
  int gid = get_global_id(0);
  int total = N * C * 100;
  if (gid >= total) return;
  int n = gid / (C * 100);
  int rem = gid % (C * 100);
  int ch = rem / 100;
  int pos = rem % 100;
  float s = g[ch] / sqrt(var[ch] + 1e-5f);
  float o = beta[ch] - mean[ch] * s;
  float v = in[gid] * s + o;
  out[gid] = v > 0.0f ? v : 0.0f;
}

// 融合 kernel：conv3x3 + BN + ReLU（一步出结果，省一次全局内存往返）
__kernel void conv3bnr(__global const float* in, __global const float* w,
                       __global const float* b, __global const float* g,
                       __global const float* beta, __global const float* mean,
                       __global const float* vr, __global float* out,
                       int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  int r = pos / 10, c = pos % 10;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) {
    for (int kr = 0; kr < 3; kr++) {
      int rr = r + kr - 1;
      if (rr < 0 || rr >= 10) continue;
      for (int kc = 0; kc < 3; kc++) {
        int cc = c + kc - 1;
        if (cc < 0 || cc >= 10) continue;
        sum += in[(n*Cin + ic)*100 + rr*10 + cc] * w[((oc*Cin + ic)*3 + kr)*3 + kc];
      }
    }
  }
  float s = g[oc] / sqrt(vr[oc] + 1e-5f);
  float o = beta[oc] - mean[oc] * s;
  float v = sum * s + o;
  out[gid] = v > 0.0f ? v : 0.0f;
}

// 融合 kernel：conv3x3 + BN（无 ReLU，残差块第二步用）
__kernel void conv3bn(__global const float* in, __global const float* w,
                      __global const float* b, __global const float* g,
                      __global const float* beta, __global const float* mean,
                      __global const float* vr, __global float* out,
                      int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  int r = pos / 10, c = pos % 10;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) {
    for (int kr = 0; kr < 3; kr++) {
      int rr = r + kr - 1;
      if (rr < 0 || rr >= 10) continue;
      for (int kc = 0; kc < 3; kc++) {
        int cc = c + kc - 1;
        if (cc < 0 || cc >= 10) continue;
        sum += in[(n*Cin + ic)*100 + rr*10 + cc] * w[((oc*Cin + ic)*3 + kr)*3 + kc];
      }
    }
  }
  float s = g[oc] / sqrt(vr[oc] + 1e-5f);
  float o = beta[oc] - mean[oc] * s;
  out[gid] = sum * s + o;
}

// 残差相加 + ReLU（通道主序）
__kernel void addrelu(__global const float* a, __global const float* b, __global float* out, int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  float v = a[gid] + b[gid];
  out[gid] = v > 0.0f ? v : 0.0f;
}

// 转置：通道主序 (N,C,100) ↔ token 主序 (N,100,C)
__kernel void transpose(__global const float* in, __global float* out, int N, int C) {
  int gid = get_global_id(0);
  int total = N * C * 100;
  if (gid >= total) return;
  int n = gid / (C * 100);
  int rem = gid % (C * 100);
  int c = rem / 100;
  int pos = rem % 100;
  out[(n*100 + pos)*C + c] = in[gid];
}

// 分头转置：Kh (N,H,NP,HD) → KhT (N,H,HD,NP)
__kernel void transpose4(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * HEADS * 100 * HD;
  if (gid >= total) return;
  int n = gid / (HEADS * 100 * HD);
  int rem = gid % (HEADS * 100 * HD);
  int h = rem / (100 * HD);
  rem = rem % (100 * HD);
  int t = rem / HD;
  int d = rem % HD;
  out[((n*HEADS + h)*HD + d)*100 + t] = in[gid];
}

// token 主序 → 通道主序：in (N,100,DM) → out (N,DM,100)
__kernel void t2c(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * 100 * DM;
  if (gid >= total) return;
  int n = gid / (100 * DM);
  int rem = gid % (100 * DM);
  int t = rem / DM;
  int d = rem % DM;
  out[(n*DM + d)*100 + t] = in[gid];
}

// 批量矩阵乘: C[n,m,k] = sum_p A[n,m,p]*B[p,k] + bias[k]
// 数据 (N, M, P) x (P, K) -> (N, M, K)
__kernel void matmul(__global const float* A, __global const float* B,
                     __global const float* bias, __global float* C,
                     int N, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = N * M * K;
  if (gid >= total) return;
  int n = gid / (M * K);
  int rem = gid % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = bias ? bias[k] : 0.0f;
  for (int p = 0; p < P; p++) sum += A[(n*M + m)*P + p] * B[p*K + k];
  C[gid] = sum;
}

// 批量矩阵乘（B 每局面独立）: C[n,m,k] = sum_p A[n,m,p]*B[n,p,k]
__kernel void matmulB(__global const float* A, __global const float* B,
                      __global float* C, int N, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = N * M * K;
  if (gid >= total) return;
  int n = gid / (M * K);
  int rem = gid % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = 0.0f;
  for (int p = 0; p < P; p++) sum += A[(n*M + m)*P + p] * B[(n*P + p)*K + k];
  C[gid] = sum;
}

// 分头批量矩阵乘: C[n,h,m,k] = sum_p A[n,h,m,p]*B[n,h,p,k]
__kernel void matmulBH(__global const float* A, __global const float* B,
                       __global float* C, int N, int H, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = N * H * M * K;
  if (gid >= total) return;
  int n = gid / (H * M * K);
  int rem = gid % (H * M * K);
  int h = rem / (M * K);
  rem = rem % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = 0.0f;
  for (int p = 0; p < P; p++) sum += A[((n*H + h)*M + m)*P + p] * B[((n*H + h)*P + p)*K + k];
  C[gid] = sum;
}

// 重排 (N,100,DM) → (N,H,100,HD)：分头
__kernel void reshapeHead(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * 100 * DM;
  if (gid >= total) return;
  int n = gid / (100 * DM);
  int rem = gid % (100 * DM);
  int t = rem / DM;
  int d = rem % DM;
  int h = d / HD;
  int dd = d % HD;
  out[((n*HEADS + h)*100 + t)*HD + dd] = in[gid];
}

// 还原 (N,H,100,HD) → (N,100,DM)
__kernel void reshapeHeadInv(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * 100 * DM;
  if (gid >= total) return;
  int n = gid / (100 * DM);
  int rem = gid % (100 * DM);
  int t = rem / DM;
  int d = rem % DM;
  int h = d / HD;
  int dd = d % HD;
  out[gid] = in[((n*HEADS + h)*100 + t)*HD + dd];
}

// 行 softmax：输入 (N, ROWS, COLS)，每行 COLS 个值；scale 用于 QK^T 缩放
__kernel void softmax(__global const float* in, __global float* out, int N, int ROWS, int COLS, float scale) {
  int row = get_global_id(0);
  int total = N * ROWS;
  if (row >= total) return;
  int n = row / ROWS;
  int r = row % ROWS;
  const __global float* x = in + (n*ROWS + r)*COLS;
  __global float* y = out + (n*ROWS + r)*COLS;
  float mx = -1e30f;
  for (int i = 0; i < COLS; i++) if (x[i] > mx) mx = x[i];
  float sum = 0.0f;
  for (int i = 0; i < COLS; i++) { float e = exp((x[i] - mx) / scale); y[i] = e; sum += e; }
  for (int i = 0; i < COLS; i++) y[i] /= sum;
}

// LayerNorm：每行 DIM 维
__kernel void layernorm(__global const float* in, __global const float* g,
                        __global const float* beta, __global float* out,
                        int N, int ROWS, int DIM) {
  int row = get_global_id(0);
  int total = N * ROWS;
  if (row >= total) return;
  int n = row / ROWS;
  int r = row % ROWS;
  const __global float* x = in + (n*ROWS + r)*DIM;
  __global float* y = out + (n*ROWS + r)*DIM;
  float mean = 0.0f;
  for (int i = 0; i < DIM; i++) mean += x[i];
  mean /= DIM;
  float vr = 0.0f;
  for (int i = 0; i < DIM; i++) vr += (x[i] - mean)*(x[i] - mean);
  vr /= DIM;
  float inv = 1.0f / sqrt(vr + 1e-5f);
  for (int i = 0; i < DIM; i++) y[i] = (x[i] - mean) * inv * g[i] + beta[i];
}

// ReLU（token 主序）
__kernel void relu_t(__global const float* in, __global float* out, int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  float v = in[gid];
  out[gid] = v > 0.0f ? v : 0.0f;
}

// 残差相加（token 主序，无 ReLU）
__kernel void add_t(__global const float* a, __global const float* b, __global float* out, int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  out[gid] = a[gid] + b[gid];
}

/* ===== v2 新增 kernel：MANO / GRN / rpb ===== */
// GRN：out[c,pos] = x + γ_c * x / (sqrt(Σ_pos x²) + 1e-6)，通道主序 (N,C,100)
__kernel void grn(__global const float* in, __global const float* gamma, __global float* out, int N, int C) {
  int gid = get_global_id(0);
  int total = N * C * 100;
  if (gid >= total) return;
  int n = gid / (C * 100);
  int rem = gid % (C * 100);
  int ch = rem / 100;
  const __global float* x = in + (n * C + ch) * 100;
  float ss = 0.0f;
  for (int p = 0; p < 100; p++) ss += x[p] * x[p];
  float inv = 1.0f / (sqrt(ss) + 1e-6f);
  float g = gamma[ch];
  for (int p = 0; p < 100; p++) out[(n * C + ch) * 100 + p] = x[p] + g * x[p] * inv;
}
// 就地版 GRN（写回同 buffer）
__kernel void grn_inplace(__global float* x, __global const float* gamma, int N, int C) {
  int gid = get_global_id(0);
  int total = N * C;
  if (gid >= total) return;
  int n = gid / C, ch = gid % C;
  __global float* row = x + (n * C + ch) * 100;
  float ss = 0.0f;
  for (int p = 0; p < 100; p++) ss += row[p] * row[p];
  float f = 1.0f + gamma[ch] / (sqrt(ss) + 1e-6f);
  for (int p = 0; p < 100; p++) row[p] *= f;
}

// MANO 降采样 conv 2×2 stride2：in (N,C,Hin,Win) → out (N,Cout,Ho,Wo)，W 布局 (out,in,2,2)
__kernel void conv2s2(__global const float* in, __global const float* w, __global const float* bias,
                      __global float* out, int N, int Cin, int Cout, int Hin, int Win, int Ho, int Wo) {
  int gid = get_global_id(0);
  int total = N * Cout * Ho * Wo;
  if (gid >= total) return;
  int n = gid / (Cout * Ho * Wo);
  int rem = gid % (Cout * Ho * Wo);
  int oc = rem / (Ho * Wo);
  int pos = rem % (Ho * Wo);
  int r = pos / Wo, c = pos % Wo;
  float sum = bias[oc];
  for (int ic = 0; ic < Cin; ic++) {
    const __global float* ip = in + ((n * Cin + ic) * Hin + r * 2) * Win + c * 2;
    const __global float* wp = w + ((oc * Cin + ic) * 2) * 2;   // [kh*2+kw]
    for (int kh = 0; kh < 2; kh++) for (int kw = 0; kw < 2; kw++)
      sum += ip[kh * Win + kw] * wp[kh * 2 + kw];
  }
  out[(n * Cout + oc) * Ho * Wo + pos] = sum;
}

// MANO 上采样 ConvTranspose2d 2×2 stride2：in (N,Cin,Hin,Win) → out (N,Cout,Ho,Wo)，W 布局 (in,out,2,2)
// 原生尺寸 (Hin-1)*2+2 × (Win-1)*2+2；目标更大时底/右留零（scatter 越界丢弃）
__kernel void convT2s2(__global const float* in, __global const float* w, __global const float* bias,
                       __global float* out, int N, int Cin, int Cout, int Hin, int Win, int Ho, int Wo) {
  int gid = get_global_id(0);
  int total = N * Cout * Ho * Wo;
  if (gid >= total) return;
  int n = gid / (Cout * Ho * Wo);
  int rem = gid % (Cout * Ho * Wo);
  int oc = rem / (Ho * Wo);
  int pos = rem % (Ho * Wo);
  int orr = pos / Wo, occ = pos % Wo;
  float sum = bias[oc];
  // out[(ic,o,kh,kw)] 布局：遍历所有 ic 及产生 (orr,occ) 的 (i,j,kh,kw)
  for (int ic = 0; ic < Cin; ic++) {
    for (int kh = 0; kh < 2; kh++) {
      int i = (orr - kh) >> 1; if (i < 0 || i >= Hin) continue; if (((orr - kh) & 1)) continue;
      for (int kw = 0; kw < 2; kw++) {
        int j = (occ - kw) >> 1; if (j < 0 || j >= Win) continue; if (((occ - kw) & 1)) continue;
        sum += in[((n * Cin + ic) * Hin + i) * Win + j] * w[((ic * Cout + oc) * 2 + kh) * 2 + kw];
      }
    }
  }
  out[gid] = sum;
}

// rpb 广播加法：Sh (N*H,100,100) += rpbEff (H,100,100)
__kernel void rpb_add(__global float* Sh, __global const float* rpbEff, int N, int Hp) {
  int gid = get_global_id(0);
  int total = N * Hp * 100 * 100;
  if (gid >= total) return;
  int n = gid / (Hp * 10000);
  int rem = gid % (Hp * 10000);
  int h = rem / 10000;
  int pos = rem % 10000;
  Sh[gid] += rpbEff[h * 10000 + pos];
}

// 通用（变长 TOK）attention 重排 kernel（MANO 各级复用）
// (B,TOK,DM) -> (B,H,TOK,HD)
__kernel void reshapeHeadT(__global const float* in, __global float* out, int B, int TOK) {
  int gid = get_global_id(0);
  int total = B * TOK * DM;
  if (gid >= total) return;
  int b = gid / (TOK * DM);
  int rem = gid % (TOK * DM);
  int t = rem / DM;
  int d = rem % DM;
  int h = d / HD;
  int dd = d % HD;
  out[((b * HEADS + h) * TOK + t) * HD + dd] = in[gid];
}
// (B,H,TOK,HD) -> (B,TOK,DM)
__kernel void reshapeHeadInvT(__global const float* in, __global float* out, int B, int TOK) {
  int gid = get_global_id(0);
  int total = B * TOK * DM;
  if (gid >= total) return;
  int b = gid / (TOK * DM);
  int rem = gid % (TOK * DM);
  int t = rem / DM;
  int d = rem % DM;
  int h = d / HD;
  int dd = d % HD;
  out[gid] = in[((b * HEADS + h) * TOK + t) * HD + dd];
}
// Kh (B,H,TOK,HD) -> KhT (B,H,HD,TOK)
__kernel void transpose4T(__global const float* in, __global float* out, int B, int TOK) {
  int gid = get_global_id(0);
  int total = B * HEADS * TOK * HD;
  if (gid >= total) return;
  int b = gid / (HEADS * TOK * HD);
  int rem = gid % (HEADS * TOK * HD);
  int h = rem / (TOK * HD);
  rem = rem % (TOK * HD);
  int t = rem / HD;
  int d = rem % HD;
  out[((b * HEADS + h) * HD + d) * TOK + t] = in[gid];
}
// 分头批量矩阵乘（变长）：C[b,h,m,k] = sum_p A[b,h,m,p]*B[b,h,p,k]
__kernel void matmulBHT(__global const float* A, __global const float* B,
                        __global float* C, int Bn, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = Bn * HEADS * M * K;
  if (gid >= total) return;
  int b = gid / (HEADS * M * K);
  int rem = gid % (HEADS * M * K);
  int h = rem / (M * K);
  rem = rem % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = 0.0f;
  for (int p = 0; p < P; p++) sum += A[((b * HEADS + h) * M + m) * P + p] * B[((b * HEADS + h) * P + p) * K + k];
  C[gid] = sum;
}

// 变长转置：(N,C,TOK) → (N,TOK,C)（MANO 各级共用，TOK=100/25/4）
__kernel void transpose_ct(__global const float* in, __global float* out, int N, int C, int TOK) {
  int gid = get_global_id(0);
  int total = N * C * TOK;
  if (gid >= total) return;
  int n = gid / (C * TOK);
  int rem = gid % (C * TOK);
  int c = rem / TOK;
  int p = rem % TOK;
  out[(n * TOK + p) * C + c] = in[gid];
}
// 变长转置：(N,TOK,C) → (N,C,TOK)
__kernel void transpose_tc(__global const float* in, __global float* out, int N, int C, int TOK) {
  int gid = get_global_id(0);
  int total = N * TOK * C;
  if (gid >= total) return;
  int n = gid / (TOK * C);
  int rem = gid % (TOK * C);
  int p = rem / C;
  int c = rem % C;
  out[(n * C + c) * TOK + p] = in[gid];
}
// MANO 融合窗口注意力：Q/K/V token 主序 (B,TOK,DM)；窗口 WIN×WIN（ROWS/COLS 被 WIN 整除）。
// WIN=5：L0=10×10 切 4 窗、L1=5×5 / L2=2×2 单窗=全图。每线程算一个 (b,t,h)，与 cnn.js manoAttn 同构。
// WIN>=max(ROWS,COLS)（flags[3]=0，窗口关闭）：r0=(r/WIN)*WIN=0 且 rr>=ROWS 不触发 → 退化为单窗全图。
// scores 容量取 MANO_WINMAX²（10×10=100），兼容两档 WIN。
__kernel void mano_attn(__global const float* Q, __global const float* K, __global const float* V,
                        __global float* Out, int Bn, int TOK, int ROWS, int COLS, int WIN) {
  int gid = get_global_id(0);
  int total = Bn * TOK * HEADS;
  if (gid >= total) return;
  int b = gid / (TOK * HEADS);
  int rem = gid % (TOK * HEADS);
  int t = rem / HEADS;
  int h = rem % HEADS;
  int r = t / COLS, c = t % COLS;
  int r0 = (r / WIN) * WIN, c0 = (c / WIN) * WIN;
  const float inv = 1.0f / sqrt((float)HD);
  const __global float* qv = Q + ((b * TOK + t) * DM + h * HD);
  float scores[MANO_WINMAX * MANO_WINMAX];
  float mx = -1e30f;
  int cnt = 0;
  for (int i = 0; i < WIN; i++) {
    for (int j = 0; j < WIN; j++) {
      int rr = r0 + i, cc = c0 + j;
      if (rr >= ROWS || cc >= COLS) continue;
      const __global float* kv = K + ((b * TOK + rr * COLS + cc) * DM + h * HD);
      float s = 0.0f;
      for (int d = 0; d < HD; d++) s += qv[d] * kv[d];
      s *= inv;
      scores[cnt++] = s;
      if (s > mx) mx = s;
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < cnt; i++) { float e = exp(scores[i] - mx); scores[i] = e; sum += e; }
  __global float* ov = Out + ((b * TOK + t) * DM + h * HD);
  for (int d = 0; d < HD; d++) ov[d] = 0.0f;
  int idx = 0;
  for (int i = 0; i < WIN; i++) {
    for (int j = 0; j < WIN; j++) {
      int rr = r0 + i, cc = c0 + j;
      if (rr >= ROWS || cc >= COLS) continue;
      float wgt = scores[idx++] / sum;
      const __global float* vv = V + ((b * TOK + rr * COLS + cc) * DM + h * HD);
      for (int d = 0; d < HD; d++) ov[d] += wgt * vv[d];
    }
  }
}
