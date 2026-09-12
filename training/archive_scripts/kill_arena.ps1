chcp 65001 > $null
# 杀对战相关进程（两服务端树 + 裁判），保留其他
$judge = Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'match_arena' }
foreach ($j in $judge) { Stop-Process -Id $j.ProcessId -Force; Write-Output ("killed judge " + $j.ProcessId) }
$srv = Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'server\.js' }
foreach ($s in $srv) { Stop-Process -Id $s.ProcessId -Force; Write-Output ("killed server " + $s.ProcessId) }
Start-Sleep 1
Write-Output ("ports: " + (Get-NetTCPConnection -LocalPort 8891,8892 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count)
