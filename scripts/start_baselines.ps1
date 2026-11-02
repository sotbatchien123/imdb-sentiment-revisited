$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'work_dirs\horror_frozen_bert_v1'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$pythonExe = Join-Path $projectRoot '.venv-train\Scripts\python.exe'
$scriptPath = Join-Path $projectRoot 'scripts\train_baselines.py'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Training environment missing.' }
$pauseFile = Join-Path $runRoot 'PAUSE'
if (Test-Path -LiteralPath $pauseFile) { Remove-Item -LiteralPath $pauseFile }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$proc = Start-Process -FilePath $pythonExe -ArgumentList @('-u', $scriptPath) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot "console_$stamp.log") -RedirectStandardError (Join-Path $runRoot "errors_$stamp.log") -PassThru
$proc.Id | Set-Content -LiteralPath (Join-Path $runRoot 'launcher_pid.txt')
Write-Output "Baseline process started: PID $($proc.Id). Check work_dirs\horror_frozen_bert_v1\status.json."
