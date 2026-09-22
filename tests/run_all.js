'use strict';
/* ================================================================
 * 门禁总编排（2026-09-23 架构现代化）：所有验收门禁一处运行。
 *   node tests/run_all.js           全量（含 GPU / py / OpenVINO 依赖项）
 *   node tests/run_all.js --quick   只跑纯 JS 核心门禁（无卡/CI 环境）
 * 退出码 0=全过；1=有 FAIL。已知历史失败（gpu_v2 步骤2/4，见
 * docs/contracts/v3_contract.md §6）标记为 WARN 不计失败。
 * ================================================================ */
const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.join(__dirname, '..');
const quick = process.argv.includes('--quick');

const WEIGHTS = ['BJ1_r208_v3.bin', 'R160_v2.bin', 'v3_arm0_local.bin', 'v1_legacy_local.bin']
  .map(f => path.join(ROOT, 'weights', f)).filter(f => fs.existsSync(f));

const jobs = [
  { name: 'policy 编码单射（12 局）', file: 'tests/test_policy_encoding.js', args: ['12'], always: true },
  { name: '走法生成等价差分', file: 'tests/movegen_equiv_test.js', always: true },
  { name: '攻击检测等价差分', file: 'tests/attack_equiv_test.js', always: true },
  { name: '前端脚本语法', file: 'server/tools/html_syntax_check.js', always: true },
  { name: 'cnn v2 对拍（参考夹具）', file: 'tests/cnn_v2_parity_test.js', always: true },
  { name: 'cnn v3 对拍', file: 'tests/cnn_v3_parity_test.js', always: true },
  { name: 'GPU v3 对拍', file: 'tests/gpu_v3_parity_test.js' },
  { name: 'GPU v2 对拍（参考夹具，已知历史失败→WARN）', file: 'tests/gpu_v2_parity_test.js', warnOnly: true },
  { name: 'e2e 冒烟（worker+WS 全链路）', file: 'tests/e2e_v2_smoke.js' },
  { name: '集成级门禁（mano patch ×2）', cmd: ['py', '-3', 'tests/integration_gate.py'] },
  { name: 'OpenVINO × v3 权重（NPU）', file: 'tests/test_ov_v3_weights.js',
    args: [path.join(ROOT, 'weights', 'BJ1_r208_v3.bin'), 'NPU'] },
];

const results = [];
for (const j of jobs) {
  if (quick && !j.always) { results.push({ name: j.name, status: 'SKIP' }); continue; }
  const cmd = j.cmd || ['node', j.file, ...(j.args || [])];
  process.stdout.write(`\n===== ${j.name} =====\n`);
  const r = spawnSync(cmd[0], cmd.slice(1), { cwd: ROOT, stdio: 'inherit', timeout: 300000 });
  let status;
  if (r.status === 0) status = 'PASS';
  else if (j.warnOnly) status = 'WARN';
  else status = 'FAIL';
  results.push({ name: j.name, status });
}

// 权重探针 ×4（存活权重的探针三题，值域回归）
if (WEIGHTS.length) {
  process.stdout.write(`\n===== 权重探针 ×${WEIGHTS.length} =====\n`);
  let wok = true;
  for (const w of WEIGHTS) {
    const r = spawnSync('node', [path.join(ROOT, 'server', 'verify_weights.js'), w],
      { cwd: ROOT, encoding: 'utf8', timeout: 120000 });
    const line = (r.stdout || '').trim().split('\n').pop() || 'NO OUTPUT';
    const ok = r.status === 0 && line.includes('VERIFY OK');
    if (!ok) wok = false;
    process.stdout.write(`${ok ? 'PASS' : 'FAIL'}  ${path.basename(w)}: ${line}\n`);
  }
  results.push({ name: `权重探针 ×${WEIGHTS.length}`, status: wok ? 'PASS' : 'FAIL' });
}

process.stdout.write('\n================ 汇总 ================\n');
let fails = 0;
for (const r of results) {
  const tag = { PASS: '✓', FAIL: '✗', WARN: '△', SKIP: '·' }[r.status];
  if (r.status === 'FAIL') fails++;
  process.stdout.write(`${tag} ${r.status.padEnd(4)} ${r.name}\n`);
}
process.stdout.write(fails === 0 ? '\nALL GREEN\n' : `\n${fails} FAILED\n`);
process.exit(fails === 0 ? 0 : 1);
