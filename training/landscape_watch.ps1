# chess10 landscape watcher (per-round loss landscape + timeline + retention)
# Pure-ASCII source on purpose: PS5.1 misreads BOM-less UTF-8 as ANSI, so the
# Chinese path segment is built from codepoints. Do NOT type Chinese into this file.
#
# Behaviour:
#   * polls fsf_teacher_loop.log every 60s for new "[R### DONE]" lines
#   * only processes rounds >= MINROUND (140 = first v2 round); older ones are
#     marked done immediately (their training data no longer exists)
#   * for each newly finished round: node round_pipeline.js r###
#     (= Li loss surface on CUDA ~10min -> rebuild timeline html -> retention:
#        keep newest round data only; keep newest snapshot + r139/r140 baselines)
#   * STOP flag: training\STOP.flag exists -> write END line and exit
#   * state: data\landscape_watch.state.json (processed round set)

$MINROUND = 140

$CN   = -join @([char]0x65B0,[char]0x5EFA,[char]0x6587,[char]0x4EF6,[char]0x5939)
$BASE = "D:\data\$CN\chess_game"
$DATA = Join-Path $BASE 'training\data'
$TDIR = Join-Path $BASE 'training'
$FLAG = Join-Path $BASE 'training\STOP.flag'
$PIDF = Join-Path $DATA 'landscape_watch.pid'
$DLOG = Join-Path $DATA 'fsf_teacher_loop.log'
$log  = Join-Path $DATA 'landscape_watch.log'
$stateFile = Join-Path $DATA 'landscape_watch.state.json'
$ErrorActionPreference = 'Continue'

function Log($s) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $s" }

# single-instance guard
if (Test-Path $PIDF) {
  $old = Get-Content $PIDF -ErrorAction SilentlyContinue
  if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
    Log "already running pid=$old, exit"
    exit
  }
}
$PID | Out-File $PIDF -Encoding ascii
Log "=== LANDSCAPE WATCHER START pid=$PID ==="

# load processed-round state
$done = @{}
if (Test-Path $stateFile) {
  try {
    $j = Get-Content $stateFile -Raw | ConvertFrom-Json
    foreach ($v in $j.processed) { $done[[int]$v] = $true }
    Log "state loaded: processed $($done.Count) rounds"
  } catch { Log "state load failed, starting fresh" }
}

function Save-State {
  $vals = @($done.Keys | Sort-Object)
  if ($vals.Count -gt 300) { $vals = @($vals | Select-Object -Last 300) }
  @{ processed = $vals } | ConvertTo-Json -Compress | Set-Content $stateFile -Encoding ascii
}

while ($true) {
  if (Test-Path $FLAG) {
    Log "=== LANDSCAPE WATCHER END (STOP flag) ==="
    Remove-Item $PIDF -ErrorAction SilentlyContinue
    break
  }
  try {
    if (Test-Path $DLOG) {
      $tail = Get-Content $DLOG -Tail 200 -ErrorAction SilentlyContinue
      $rounds = @()
      foreach ($line in $tail) {
        if ($line -match 'R(\d+) DONE') { $rounds += [int]$Matches[1] }
      }
      $rounds = @($rounds | Sort-Object -Unique)
      $ranOne = $false
      foreach ($r in $rounds) {
        if (-not $done.ContainsKey($r)) {
          if ($r -lt $MINROUND) {
            $done[$r] = $true
            Save-State
            Log "[R$r] pre-MINROUND, skipped"
            continue
          }
          Log "[R$r] DONE detected, running pipeline"
          $out = & node (Join-Path $TDIR 'round_pipeline.js') ("r" + $r) 2>&1
          $code = $LASTEXITCODE
          foreach ($l in @($out)) { Log "[R$r] $l" }
          if ($code -eq 0) {
            $done[$r] = $true
            Log "[R$r] PIPELINE OK"
          } elseif ($code -eq 3) {
            $done[$r] = $true
            Log "[R$r] SKIPPED (training data pruned, landscape impossible)"
          } else {
            Log "[R$r] PIPELINE FAILED code=$code (will retry next scan)"
          }
          Save-State
          $ranOne = $true
        }
        if ($ranOne) { break }   # one pipeline per scan: retention of a later round must never prune data of an earlier unprocessed round
      }
    }
  } catch { Log "ERR $_" }
  Start-Sleep -Seconds 60
}
