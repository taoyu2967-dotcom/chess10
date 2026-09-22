# launcher for landscape_watch.ps1 (Start-Process pattern: node-spawned detached
# powershell dies silently, must use Start-Process). Pure-ASCII source.
# Path hub convention (2026-09-23): CHESS10_ROOT env wins over script location.
$BASE = if ($env:CHESS10_ROOT) { $env:CHESS10_ROOT } else { Split-Path -Parent $PSScriptRoot }
$TDIR = Join-Path $BASE 'training'
Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $TDIR 'landscape_watch.ps1')) -WindowStyle Hidden
Write-Output "landscape watcher launched"
