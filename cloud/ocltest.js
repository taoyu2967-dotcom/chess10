'use strict';
// OpenCL 设备枚举冒烟：验证容器里 OpenCL 能用、gpu.js 能初始化
const gpu = require('/root/autodl-tmp/chess/server/gpu.js');
try {
  const r = gpu.init();
  console.log('OPENCL_OK device=' + gpu.getDevice() + ' batch=' + (r && r.batch));
} catch (e) {
  console.log('OPENCL_FAIL ' + (e && e.message));
  process.exit(1);
}
