$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'work_dirs\imdb_full_v1'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$pythonExe = Join-Path $projectRoot '.venv-train\Scripts\python.exe'
$scriptPath = Join-Path $projectRoot 'scripts\train_baselines.py'
$pauseFile = Join-Path $runRoot 'PAUSE'
if (Test-Path -LiteralPath $pauseFile) { Remove-Item -LiteralPath $pauseFile }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$proc = Start-Process -FilePath $pythonExe -ArgumentList @('-u', $scriptPath, '--config', 'configs/full_imdb.json') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot "console_$stamp.log") -RedirectStandardError (Join-Path $runRoot "errors_$stamp.log") -PassThru
$proc.Id | Set-Content -LiteralPath (Join-Path $runRoot 'launcher_pid.txt')
$observerRoot = Join-Path $projectRoot 'reports\full_imdb_status'
New-Item -ItemType Directory -Force -Path $observerRoot | Out-Null
Start-Process -FilePath $pythonExe -ArgumentList @('-u', (Join-Path $projectRoot 'scripts\publish_model_status.py'), '--watch') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $observerRoot "observer_$stamp.log") -RedirectStandardError (Join-Path $observerRoot "observer_errors_$stamp.log") | Out-Null
Write-Output "Full IMDb started: PID $($proc.Id). Check status_live.json and reports/full_imdb_report.md."
