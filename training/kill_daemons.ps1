# 重启守护：清掉所有旧实例（含误起的前台实例），用修复后的脚本重启
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'daily_train' } | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force; Write-Output ("killed daemon " + $_.ProcessId)
}
Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'selfplay_fsf_teacher' } | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force; Write-Output ("killed teacher node " + $_.ProcessId)
}
Get-Process fairy-stockfish -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2
Write-Output ("daemons left: " + (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'daily_train' } | Measure-Object).Count)
