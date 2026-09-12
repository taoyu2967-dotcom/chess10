'use strict';
// OpenVINO 混合推理桥客户端：spawn py ov_bridge_server.py，二进制帧协议
const { spawn } = require('child_process');

class OvBridge {
  constructor(script, args = []) {
    this.proc = spawn('py', [script, ...args], { stdio: ['pipe', 'pipe', 'pipe'] });
    this.chunks = [];
    this.bufLen = 0;
    this.waiters = [];
    this.dead = false;
    this.ready = new Promise((res, rej) => {
      const to = setTimeout(() => rej(new Error('OV bridge 90s 未就绪')), 90000);
      this._readyRes = () => { clearTimeout(to); res(); };
      this._readyRej = (e) => { clearTimeout(to); rej(e); };
    });
    this.proc.stderr.on('data', d => {
      const s = d.toString();
      if (s.includes('READY') && this._readyRes) { const f = this._readyRes; this._readyRes = null; f(); }
      else process.stderr.write('[OV] ' + s);
    });
    this.proc.stdout.on('data', chunk => { this.chunks.push(chunk); this.bufLen += chunk.length; this._drain(); });
    this.proc.on('exit', () => {
      this.dead = true;
      if (this._readyRej) { const f = this._readyRej; this._readyRej = null; f(new Error('bridge 启动即退出')); }
      this.waiters.forEach(w => w.rej(new Error('OV bridge 已退出')));
      this.waiters = [];
    });
    this.proc.on('error', e => { this.dead = true; if (this._readyRej) { this._readyRej(e); } });
  }

  _drain() {
    while (this.waiters.length) {
      if (this.bufLen < 8) return;
      const buf = this.chunks.length === 1 ? this.chunks[0] : Buffer.concat(this.chunks, this.bufLen);
      const n = buf.readInt32LE(0);
      const totalQ = buf.readInt32LE(4);
      if (n <= 0) {
        const rest = buf.subarray(8);
        this.chunks = rest.length ? [rest] : [];
        this.bufLen = rest.length;
        continue;
      }
      const need = 8 + n * 4 + totalQ * 4;
      if (buf.length < need) { this.chunks = [buf]; this.bufLen = buf.length; return; }
      const frame = buf.subarray(0, need);
      const rest = buf.subarray(need);
      this.chunks = rest.length ? [rest] : [];
      this.bufLen = rest.length;
      // 快速解析：两次 memcpy
      const values = new Float32Array(n);
      frame.copy(Buffer.from(values.buffer), 0, 8, 8 + n * 4);
      const logits = new Float32Array(totalQ);
      if (totalQ > 0) frame.copy(Buffer.from(logits.buffer), 0, 8 + n * 4, need);
      const w = this.waiters.shift();
      w.res({ values, logits });
    }
  }

  // boards: Int32Array(N*107)；queriesList: Array<Int32Array>（每局面的查询索引，ch*100+fromPos）
  evalBatch(boards, N, queriesList) {
    if (this.dead) return Promise.reject(new Error('OV bridge 已退出'));
    let totalQ = 0;
    for (const q of queriesList) totalQ += q.length;
    const req = Buffer.allocUnsafe(8 + N * 107 * 4 + (N + totalQ) * 4);
    req.writeInt32LE(N, 0);
    req.writeInt32LE(totalQ, 4 + N * 107 * 4);
    Buffer.from(boards.buffer, boards.byteOffset, N * 107 * 4).copy(req, 4);
    let off = 8 + N * 107 * 4;
    for (const q of queriesList) {
      req.writeInt32LE(q.length, off); off += 4;
      for (let i = 0; i < q.length; i++) { req.writeInt32LE(q[i], off); off += 4; }
    }
    this.proc.stdin.write(req);
    return new Promise((res, rej) => this.waiters.push({ res, rej }));
  }

  kill() {
    try { this.proc.stdin.end(); } catch (e) {}
    try { this.proc.kill(); } catch (e) {}
  }
}

// 多流水线桥：KataGo 式异构设备并行（每条流水线 = 独立桥进程 + 按速度分配批量）
// streams: [{ script, args, share }]，share 为吞吐权重（如 iGPU 0.75 / NPU 0.25）
class MultiBridge {
  constructor(streams) {
    this.streams = streams.map(s => ({ bridge: new OvBridge(s.script, s.args), share: s.share }));
  }
  async ready() { await Promise.all(this.streams.map(s => s.bridge.ready)); }
  async evalBatch(boards, N, queriesList) {
    if (this.streams.length === 1 || N < 16) return this.streams[0].bridge.evalBatch(boards, N, queriesList);
    const total = this.streams.reduce((a, s) => a + s.share, 0);
    const jobs = [];
    let off = 0;
    for (let i = 0; i < this.streams.length; i++) {
      const n = (i === this.streams.length - 1) ? N - off : Math.max(0, Math.round(N * this.streams[i].share / total));
      if (n > 0) jobs.push({
        off, n,
        p: this.streams[i].bridge.evalBatch(
          boards.subarray(off * 107, (off + n) * 107), n, queriesList.slice(off, off + n)),
      });
      off += n;
    }
    const res = await Promise.all(jobs.map(j => j.p));
    const values = new Float32Array(N);
    let totalQ = 0;
    for (const r of res) totalQ += r.logits.length;
    const logits = new Float32Array(totalQ);
    let vOff = 0, lOff = 0;
    for (const r of res) {
      values.set(r.values, vOff); vOff += r.values.length;
      logits.set(r.logits, lOff); lOff += r.logits.length;
    }
    return { values, logits };
  }
  kill() { this.streams.forEach(s => s.bridge.kill()); }
}

module.exports = { OvBridge, MultiBridge };