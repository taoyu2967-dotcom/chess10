chcp 65001 > $null
$j = Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'match_arena' }
foreach ($p in $j) { Stop-Process -Id $p.ProcessId -Force; Write-Output ("killed judge " + $p.ProcessId) }
if (-not $j) { Write-Output "judge already gone" }
