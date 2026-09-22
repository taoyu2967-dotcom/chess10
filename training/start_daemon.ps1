# daemon launcher (pure ASCII source)
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File start_daemon.ps1
# Path hub convention (2026-09-23): training dir from script location; CHESS10_ROOT env wins.
$T    = if ($env:CHESS10_ROOT) { Join-Path $env:CHESS10_ROOT 'training' } else { $PSScriptRoot }
$PIDF = Join-Path $T 'data\daemon.pid'
# already running?
if (Test-Path -LiteralPath $PIDF) {
  $old = Get-Content -LiteralPath $PIDF -ErrorAction SilentlyContinue
  if ($old) {
    $alive = Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue
    if ($alive) { Write-Output "daemon already running pid=$old"; exit 0 }
  }
}
$p = Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $T 'daily_train.ps1') -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $PIDF -Value $p.Id
Write-Output "daemon launched pid=$($p.Id)"
