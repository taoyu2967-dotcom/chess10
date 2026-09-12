# -*- coding: utf-8 -*-
# convert_model 分 batch 计时诊断：定位 b256 导出卡点
import faulthandler, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN  # noqa: E402

net = build_net(os.path.join(HERE, '..', 'weights_ov.bin')).eval()

for B in (32, 128, 256):
    faulthandler.dump_traceback_later(90, exit=False)
    t0 = time.time()
    print(f'--- convert B={B} start', flush=True)
    m = ov.convert_model(net.eval(), example_input=torch.zeros(B, C_IN, 10, 10),
                         input=(B, C_IN, 10, 10))
    print(f'--- convert B={B} OK in {time.time()-t0:.1f}s, outs={len(m.outputs)}', flush=True)
    faulthandler.cancel_dump_traceback_later()
print('ALL DONE', flush=True)
