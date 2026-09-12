# FSF teacher training loop v3 (persistent, pure-ASCII source to survive PS5.1 ANSI parsing)
# Chinese path segment is built from codepoints: 新建文件夹
$CN = -join @([char]0x65B0, [char]0x5EFA, [char]0x6587, [char]0x4EF6, [char]0x5939)
$Hours = 4.5; if ($env:CHESS10_FSFHOURS)  { $Hours = [double]$env:CHESS10_FSFHOURS }
$GAMES = 120; if ($env:CHESS10_FSFGAMES)  { $GAMES = [int]$env:CHESS10_FSFGAMES }
$MT = 150;    if ($env:CHESS10_FSFMT)     { $MT = [int]$env:CHESS10_FSFMT }
$EPS = 4;     if ($env:CHESS10_FSFEPOCHS) { $EPS = [int]$env:CHESS10_FSFEPOCHS }
$START_ROUND = 60
$ErrorActionPreference = 'Continue'
$BASE      = "D:\data\$CN\chess_game"
$serverDir = Join-Path $BASE 'server'
$dataDir   = Join-Path $BASE 'training\data'
$ovDir     = Join-Path $BASE 'ov_train'
$snapDir   = Join-Path $dataDir 'snapshots'
New-Item -ItemType Directory -Force -Path $snapDir | Out-Null
$log = Join-Path $dataDir 'fsf_teacher_loop.log'
$deadline = (Get-Date).AddHours($Hours)
$round = $START_ROUND - 1
Add-Content $log "=== FSF TEACHER LOOP START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') hours=$Hours games=$GAMES mt=$MT ==="
while ((Get-Date) -lt $deadline) {
  $round++
  $t0 = Get-Date
  $remainMin = ($deadline - (Get-Date)).TotalMinutes
  if ($remainMin -lt 15) { break }
  # 1) teacher data generation (14-core parallel)
  $spFile = Join-Path $dataDir ("fsf_r{0}.out" -f $round)
  Push-Location $serverDir
  cmd /c "node selfplay_fsf_teacher.js $GAMES $MT `"$dataDir\r$round`" > `"$spFile`" 2>&1"
  $spCode = $LASTEXITCODE
  Pop-Location
  $spTxt = ''
  if (Test-Path $spFile) { $spTxt = (Get-Content $spFile -Raw -ErrorAction SilentlyContinue) }
  $spOne = ($spTxt -replace "[\r\n]+", " | ").Trim()
  Add-Content $log "[R$round SP $(Get-Date -Format 'HH:mm:ss') code=$spCode] $spOne"
  if ($spCode -ne 0 -or -not $spTxt) { Add-Content $log "[R$round] teacher gen FAILED, skip"; Start-Sleep 10; continue }
  # 2) GPU training
  $trFile = Join-Path $dataDir ("tr_r{0}.out" -f $round)
  cmd /c "py `"$ovDir\torch_ov_train.py`" $EPS 1e-4 $round > `"$trFile`" 2>&1"
  $trCode = $LASTEXITCODE
  $trTxt = ''
  if (Test-Path $trFile) { $trTxt = (Get-Content $trFile -Raw -ErrorAction SilentlyContinue) }
  $trOne = ($trTxt -replace "[\r\n]+", " | ").Trim()
  Add-Content $log "[R$round TR $(Get-Date -Format 'HH:mm:ss') code=$trCode] $trOne"
  if ($trCode -ne 0) { Add-Content $log "[R$round] train FAILED, keep old weights"; Start-Sleep 10; continue }
  # 3) verify
  Push-Location $serverDir
  $vfFile = Join-Path $dataDir ("vf_r{0}.out" -f $round)
  cmd /c "node verify_weights.js weights_ov_new.bin > `"$vfFile`" 2>&1"
  $vfCode = $LASTEXITCODE
  Pop-Location
  $vfTxt = ''
  if (Test-Path $vfFile) { $vfTxt = (Get-Content $vfFile -Raw -ErrorAction SilentlyContinue) }
  if ($vfCode -ne 0) { Add-Content $log "[R$round] VERIFY FAILED: $($vfTxt.Trim()), keep old weights"; Start-Sleep 10; continue }
  Add-Content $log "[R$round VF] $($vfTxt.Trim())"
  # 4) promote + snapshot
  Copy-Item (Join-Path $serverDir 'weights_ov_new.bin') (Join-Path $serverDir 'weights_ov.bin') -Force
  Copy-Item (Join-Path $serverDir 'weights_ov.bin') (Join-Path $snapDir ("r{0}.bin" -f $round)) -Force
  Add-Content $log "[R$round DONE] secs=$([math]::Round(((Get-Date)-$t0).TotalSeconds)) snapshot=r$round.bin"
}
Add-Content $log "=== FSF TEACHER LOOP END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') rounds_completed=$($round - $START_ROUND + 1) ==="
