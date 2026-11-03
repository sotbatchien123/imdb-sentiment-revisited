@echo off
cd /d "%~dp0.."
powershell -NoProfile -Command "Get-Content -LiteralPath work_dirs\imdb_full_v1\status_live.json"
pause
