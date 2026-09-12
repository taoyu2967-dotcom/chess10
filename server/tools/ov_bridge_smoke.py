# -*- coding: utf-8 -*-
# OV 桥冒烟驱动：起桥 -> 等READY -> 发 2 个全零盘 -> 校验响应帧 -> 打印 MODE/耗时
import os, struct, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import PCH, N_POS  # noqa: E402
POL_SIZE = PCH * N_POS           # v3 策略宽度 = 160 * 100 = 16000
SRV = os.path.normpath(os.path.join(HERE, '..'))
BRIDGE = os.path.join(SRV, 'ov_bridge_server_v2.py')
WEIGHTS = os.path.join(SRV, 'weights_ov.bin')

proc = subprocess.Popen(
    [sys.executable, BRIDGE, '--weights', WEIGHTS, '--batch', '32'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

ready = threading.Event()
mode_lines = []

def pump_stderr():
    for raw in proc.stderr:
        line = raw.decode('utf-8', 'replace').rstrip()
        print('[bridge-stderr]', line, flush=True)
        if line.strip() == 'READY':
            ready.set()
        if '[OV] MODE' in line:
            mode_lines.append(line)

threading.Thread(target=pump_stderr, daemon=True).start()

t0 = time.time()
if not ready.wait(timeout=420):
    print('SMOKE FAIL: READY timeout after 420s, killing')
    proc.kill()
    sys.exit(1)
print(f'SMOKE: READY in {time.time()-t0:.1f}s; mode={mode_lines}')

# 发 2 个全零盘（网络前向对全零输入也应是有限值）
boards = np.zeros((2, 107), dtype=np.int32)
proc.stdin.write(struct.pack('<i', 2) + boards.tobytes())
proc.stdin.flush()

hdr = proc.stdout.read(4)
assert hdr and len(hdr) == 4, 'no response header'
n = struct.unpack('<i', hdr)[0]
body = proc.stdout.read(n * 4 + n * POL_SIZE * 4)
assert len(body) == n * 4 + n * POL_SIZE * 4, f'short body {len(body)}'
vals = np.frombuffer(body[:n * 4], dtype=np.float32)
pols = np.frombuffer(body[n * 4:], dtype=np.float32)
print(f'SMOKE: n={n} values={vals} pol_finite={np.isfinite(pols).all()} pol_shape={pols.shape}')
ok = np.isfinite(vals).all() and np.isfinite(pols).all() and pols.shape == (2 * POL_SIZE,)
print('SMOKE PASS' if ok else 'SMOKE FAIL: non-finite outputs')
proc.stdin.close()
try: proc.wait(timeout=10)
except Exception: proc.kill()
sys.exit(0 if ok else 1)
