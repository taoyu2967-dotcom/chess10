@echo off
chcp 65001 >nul
title 国际象棋 · 实时胜率
echo ============================================
echo   国际象棋 · 实时胜率（Stockfish 引擎）
echo ============================================
echo.
echo 正在启动本地服务器 (http://localhost:8123) ...
echo 稍后浏览器将自动打开，关闭本窗口即停止游戏服务。
echo.

cd /d "%~dp0"

rem 找到可用的 Python
set PY=
where py >nul 2>nul && set PY=py
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
    echo [错误] 未找到 Python。请先安装 Python 3，或手动运行：
    echo        python -m http.server 8123
    pause
    exit /b 1
)

rem 启动服务器
start "" http://localhost:8123/index.html
%PY% -m http.server 8123
