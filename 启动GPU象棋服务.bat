@echo off
title 10x10 Chess GPU Server Launcher
chcp 65001 >nul
cd /d "%~dp0server"

echo ============================================
echo  10x10 Chess - GPU Server (MCTS + CNN + OpenCL)
echo ============================================
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found. Please install Node.js first:
  echo         https://nodejs.org/
  pause
  exit /b 1
)

if not exist node_modules (
  echo First run detected - installing dependencies...
  call npm install --no-fund --no-audit
  if errorlevel 1 (
    echo [ERROR] npm install failed. Check your network.
    pause
    exit /b 1
  )
  echo.
)

netstat -ano | findstr ":8787 " | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo Port 8787 already in use - GPU server may already be running.
  timeout /t 2 /nobreak >nul
  start "" "http://localhost:8787"
  echo.
  echo Game opened in browser. Close this window if done.
  timeout /t 3 /nobreak >nul
  exit /b 0
)

echo Starting GPU server in a new window...
start "Chess10 GPU Server" cmd /k "title Chess10 GPU Server && node server.js"

echo Waiting for server to initialize GPU...
timeout /t 5 /nobreak >nul

start "" "http://localhost:8787"
echo.
echo Browser opened: http://localhost:8787
echo GPU server is running in the "Chess10 GPU Server" window.
echo Close that window to stop the server.
timeout /t 5 /nobreak >nul
