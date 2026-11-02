@echo off
cd /d "%~dp0"
".venv-train\Scripts\python.exe" -u scripts\train_reproduction.py
pause
