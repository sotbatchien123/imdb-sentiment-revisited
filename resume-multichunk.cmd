@echo off
cd /d "%~dp0"
".venv-train\Scripts\python.exe" -u scripts\train_multichunk.py
pause
