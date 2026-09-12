# chess10 daily training daemon v1 (2026-08-27)
# Pure-ASCII source on purpose: PS5.1 misreads BOM-less UTF-8 as ANSI, so the
# Chinese path segment is built from codepoints. Do NOT type Chinese into this file.
#
# Behaviour:
#   * runs rounds forever: teacher gen (14-core) -> CUDA train -> verify -> promote+snapshot
#   * sleep window 14:00-17:59 (user wants machine free 14:00-18:00)
#   * STOP flag: training\STOP.flag exists -> write END line and exit
#   * GPU self-heal: node gpu require check; on fail run tools\restore_gpu_stack.js once per round
#   * auto round numbering: max(r*_encs.f32)+1 at start
#   * data rotation: keep newest 6 rounds; snapshots keep newest 40
$CN   = -join @([char]0x65B0,[char]0x5EFA,[char]0x6587,[char]0x4EF6,[char]0x5939)
$BASE = "D:\data\$CN\chess_game"
$SRV  = Join-Path $BASE 'server'
$DATA = Join-Path $BASE 'training\data'
$OVT  = Join-Path $BASE 'ov_train'
$SNAP = Join-Path $DATA 'snapshots'
$FLAG = Join-Path $BASE 'training\STOP.flag'
$GAMES = 120; if ($env:CHESS10_FSFGAMES)  { $GAMES = [int]$env:CHESS10_FSFGAMES }
$MT    = 150; if ($env:CHESS10_FSFMT)     { $MT    = [int]$env:CHESS10_FSFMT }
$EPS   = 4;   if ($env:CHESS10_FSFEPOCHS) { $EPS   = [int]$env:CHESS10_FSFEPOCHS }
$KEEP_ROUNDS = 6; if ($env:CHESS10_KEEP_ROUNDS) { $KEEP_ROUNDS = [int]$env:CHESS10_KEEP_ROUNDS }
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $SNAP | Out-Null
$log = Join-Path $DATA 'fsf_teacher_loop.log'

function Log($s) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $s" }

# ---- auto round number ----
$round = 0
$existing = Get-ChildItem $DATA -Filter 'r*_encs.f32' -ErrorAction SilentlyContinue | ForEach-Object {
  if ($_ -match '^r(\d+)_encs\.f32$') { [int]$Matches[1] } } | Sort-Object
if ($existing) { $round = $existing[-1] }

function Test-StopFlag { Test-Path -LiteralPath $FLAG }
function Test-Blackout {
  $h = [int](Get-Date -Format 'HH')
  return ($h -ge 14 -and $h -lt 18)
}
function Test-GpuOk {
  # must check with server dir as cwd: require('./gpu') is cwd-relative
  try {
    Push-Location $SRV
    $out = & node -e "try{require('./gpu');console.log('ok')}catch(e){console.log('fail')}" 2>$null
    Pop-Location
    return ($out -join '' -match 'ok')
  } catch { try { Pop-Location } catch {}; return $false }
}
function Remove-OldRounds {
  $rs = Get-ChildItem $DATA -Filter 'r*_encs.f32' -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_ -match '^r(\d+)_encs\.f32$') { [int]$Matches[1] } } | Sort-Object
  if ($rs -and $rs.Count -gt $KEEP_ROUNDS) {
    foreach ($old in ($rs | Select-Object -First ($rs.Count - $KEEP_ROUNDS))) {
      foreach ($suf in @('_encs.f32','_pis.f32','_zs.f32')) {
        Remove-Item -LiteralPath (Join-Path $DATA ("r{0}{1}" -f $old, $suf)) -Force -ErrorAction SilentlyContinue
      }
      foreach ($pre in @('fsf_','tr_','vf_')) {
        Remove-Item -LiteralPath (Join-Path $DATA ("{0}r{1}.out" -f $pre, $old)) -Force -ErrorAction SilentlyContinue
      }
      Log "[ROTATE] removed round r$old data (keep newest $KEEP_ROUNDS)"
    }
  }
  $sn = Get-ChildItem $SNAP -Filter 'r*.bin' -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_ -match '^r(\d+)\.bin$') { [int]$Matches[1] } } | Sort-Object
  if ($sn -and $sn.Count -gt 40) {
    foreach ($old in ($sn | Select-Object -First ($sn.Count - 40))) {
      Remove-Item -LiteralPath (Join-Path $SNAP ("r{0}.bin" -f $old)) -Force -ErrorAction SilentlyContinue
    }
  }
}

Log "=== DAILY DAEMON START round_from=$(( $round + 1 )) games=$GAMES mt=$MT keep=$KEEP_ROUNDS pid=$PID ==="
while ($true) {
  if (Test-StopFlag) { Log "=== DAILY DAEMON END (STOP flag) rounds_this_run_done === "; break }
  if (Test-Blackout) { Start-Sleep -Seconds 600; continue }
  if (-not (Test-GpuOk)) {
    Log "[HEAL] GPU module check failed, running restore_gpu_stack.js"
    Push-Location $SRV
    & node (Join-Path $SRV 'tools\restore_gpu_stack.js') 2>&1 | ForEach-Object { Log "[HEAL] $_" }
    Pop-Location
    if (-not (Test-GpuOk)) { Log "[HEAL] still failing, sleep 600s"; Start-Sleep -Seconds 600; continue }
  }
  $round++
  $t0 = Get-Date
  # 1) teacher generation
  $spFile = Join-Path $DATA ("fsf_r{0}.out" -f $round)
  Push-Location $SRV
  cmd /c "node selfplay_fsf_teacher.js $GAMES $MT `"$DATA\r$round`" > `"$spFile`" 2>&1"
  $spCode = $LASTEXITCODE
  Pop-Location
  $spTxt = ''
  if (Test-Path -LiteralPath $spFile) { $spTxt = (Get-Content -LiteralPath $spFile -Raw -ErrorAction SilentlyContinue) }
  $spOne = ($spTxt -replace "[\r\n]+", " | ").Trim()
  Log "[R$round SP $(Get-Date -Format 'HH:mm:ss') code=$spCode] $spOne"
  if ($spCode -ne 0 -or -not $spTxt) { Log "[R$round] teacher gen FAILED, skip"; Start-Sleep 10; $round--; continue }
  # 2) GPU train (gate inside trainer)
  $trFile = Join-Path $DATA ("tr_r{0}.out" -f $round)
  cmd /c "py `"$OVT\torch_ov_train.py`" $EPS 1e-4 $round > `"$trFile`" 2>&1"
  $trCode = $LASTEXITCODE
  $trTxt = ''
  if (Test-Path -LiteralPath $trFile) { $trTxt = (Get-Content -LiteralPath $trFile -Raw -ErrorAction SilentlyContinue) }
  $trOne = ($trTxt -replace "[\r\n]+", " | ").Trim()
  Log "[R$round TR $(Get-Date -Format 'HH:mm:ss') code=$trCode] $trOne"
  if ($trCode -ne 0) { Log "[R$round] train FAILED (gate/fatal), keep old weights"; Start-Sleep 10; continue }
  # 3) verify
  $vfFile = Join-Path $DATA ("vf_r{0}.out" -f $round)
  Push-Location $SRV
  cmd /c "node verify_weights.js weights_ov_new.bin > `"$vfFile`" 2>&1"
  $vfCode = $LASTEXITCODE
  Pop-Location
  $vfTxt = ''
  if (Test-Path -LiteralPath $vfFile) { $vfTxt = (Get-Content -LiteralPath $vfFile -Raw -ErrorAction SilentlyContinue) }
  if ($vfCode -ne 0) { Log "[R$round] VERIFY FAILED: $($vfTxt.Trim()), keep old weights"; Start-Sleep 10; continue }
  Log "[R$round VF] $($vfTxt.Trim())"
  # 4) promote + snapshot
  Copy-Item (Join-Path $SRV 'weights_ov_new.bin') (Join-Path $SRV 'weights_ov.bin') -Force
  Copy-Item (Join-Path $SRV 'weights_ov.bin') (Join-Path $SNAP ("r{0}.bin" -f $round)) -Force
  Log "[R$round DONE] secs=$([math]::Round(((Get-Date)-$t0).TotalSeconds)) snapshot=r$round.bin"
  Remove-OldRounds
}
Log "=== DAILY DAEMON EXIT $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
