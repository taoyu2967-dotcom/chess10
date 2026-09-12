# -*- coding: utf-8 -*-
# NPU compile 耗时随 batch 的标定：决定桥默认 batch
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN  # noqa: E402

net = build_net(os.path.join(HERE, '..', 'weights_ov.bin')).eval()
core = ov.Core()
CACHE = os.path.join(HERE, '..', 'models_ov')
core.set_property({'CACHE_DIR': os.path.join(CACHE, 'cache')})

for B in (int(a) for a in (sys.argv[1:] or ['128', '64'])):
    xml = os.path.join(CACHE, f'full_v2_b{B}.xml')
    if os.path.exists(xml):
        m = core.read_model(xml)
    else:
        m = ov.convert_model(net.eval(), example_input=torch.zeros(B, C_IN, 10, 10),
                             input=(B, C_IN, 10, 10))
        ov.save_model(m, xml)
    t0 = time.time()
    try:
        compiled = core.compile_model(m, 'NPU')
        print(f'B={B}: NPU compile OK in {time.time()-t0:.1f}s', flush=True)
        # 顺手测一次推理延迟
        ireq = compiled.create_infer_request()
        x = np.zeros((B, C_IN, 10, 10), dtype=np.float32)
        ireq.infer({0: x})
        t1 = time.time()
        for _ in range(10):
            ireq.infer({0: x})
        print(f'B={B}: infer {10}次 平均 {(time.time()-t1)/10*1000:.1f}ms', flush=True)
        del ireq, compiled
    except Exception as e:
        print(f'B={B}: NPU compile FAIL {time.time()-t0:.1f}s: {str(e)[:160]}', flush=True)
print('DIAG DONE', flush=True)
