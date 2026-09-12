@echo off
chcp 65001 >nul
title 10x10 国际象棋 · 单文件版
echo   10×10 国际象棋 · 单文件版
echo   ================================
echo.
where node >nul 2>nul
if errorlevel 1 (
  echo   [!] 未检测到 Node.js，请先安装: https://nodejs.org
  pause
  exit /b 1
)
echo   启动服务: http://localhost:8787
echo   浏览器将自动打开...
start "" node chess10_server.js
timeout /t 4 /nobreak >nul
start http://localhost:8787
echo.
echo   提示:
echo     - 默认 GPU 加速需要 opencl-raub（可选）: npm install opencl-raub
echo     - 没有 GPU 也能跑（自动 CPU 回退，速度较慢）
echo     - 关闭窗口即停止服务
pause >nul
