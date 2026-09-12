# 结束所有 chess_game 相关进程（游戏服务器/worker/引擎/对战残留），不动 ZCode 自身的 MCP 服务
$killed = @()
foreach ($n in Get-CimInstance Win32_Process -Filter "Name='node.exe'") {
  if ($n.CommandLine -match 'server\.js|worker\.js|selfplay|match_|boot_all|torch') {
    Stop-Process -Id $n.ProcessId -Force -ErrorAction SilentlyContinue
    $killed += "node $($n.ProcessId)"
  }
}
foreach ($f in (Get-Process fairy-stockfish -ErrorAction SilentlyContinue)) {
  Stop-Process -Id $f.Id -Force; $killed += "fsf $($f.Id)"
}
foreach ($p in (Get-CimInstance Win32_Process -Filter "Name='python.exe'")) {
  if ($p.CommandLine -match 'ov_bridge|torch_ov_train|openvino') {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    $killed += "python $($p.ProcessId)"
  }
}
Start-Sleep 2
if ($killed.Count) { $killed | ForEach-Object { Write-Output "killed: $_" } } else { Write-Output "no chess_game processes found" }
Write-Output ("ports 8787/8891/8892 alive: " + ((Get-NetTCPConnection -LocalPort 8787,8891,8892 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count))
Write-Output ("fairy-stockfish left: " + (Get-Process fairy-stockfish -ErrorAction SilentlyContinue).Count)
