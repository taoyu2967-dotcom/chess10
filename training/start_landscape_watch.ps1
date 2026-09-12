# launcher for landscape_watch.ps1 (Start-Process pattern: node-spawned detached
# powershell dies silently, must use Start-Process). Pure-ASCII source.
$CN   = -join @([char]0x65B0,[char]0x5EFA,[char]0x6587,[char]0x4EF6,[char]0x5939)
$BASE = "D:\data\$CN\chess_game"
$TDIR = Join-Path $BASE 'training'
Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $TDIR 'landscape_watch.ps1')) -WindowStyle Hidden
Write-Output "landscape watcher launched"
