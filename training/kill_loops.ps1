chcp 65001 > $null
$loops = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'run_fsf_teacher_loop' }
foreach ($p in $loops) { Stop-Process -Id $p.ProcessId -Force; Write-Output ("killed loop " + $p.ProcessId) }
$nodes = Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'selfplay_fsf_teacher' }
foreach ($n in $nodes) { Stop-Process -Id $n.ProcessId -Force; Write-Output ("killed node " + $n.ProcessId) }
Get-Process fairy-stockfish -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2
Write-Output ("engines left: " + (Get-Process fairy-stockfish -ErrorAction SilentlyContinue).Count)
