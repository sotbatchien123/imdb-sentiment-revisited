@echo off
cd /d "%~dp0"
powershell -NoProfile -Command "Get-Content -LiteralPath work_dirs\horror_frozen_bert_v1\status.json"
pause
