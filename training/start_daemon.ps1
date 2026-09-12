# daemon launcher (ASCII source; Chinese path segment via codepoints)
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File start_daemon.ps1
$CN   = -join @([char]0x65B0,[char]0x5EFA,[char]0x6587,[char]0x4EF6,[char]0x5939)
$T    = "D:\data\$CN\chess_game\training"
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
