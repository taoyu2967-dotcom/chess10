chcp 65001 > $null
$env:CHESS10_FSFHOURS = '4.5'
$env:CHESS10_FSFGAMES = '120'
$env:CHESS10_FSFMT = '150'
$env:CHESS10_FSFEPOCHS = '4'
$loop = 'D:\data\' + [char]0x65B0 + [char]0x5EFA + [char]0x6587 + [char]0x4EF6 + [char]0x5939 + '\chess_game\training\run_fsf_teacher_loop.ps1'
if (-not (Test-Path $loop)) { Write-Output ('LOOP FILE NOT FOUND: ' + $loop); exit 1 }
Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $loop -WindowStyle Hidden
Write-Output 'LAUNCHED (bom-fixed loop)'
