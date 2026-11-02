@echo off
cd /d "%~dp0"
".venv-train\Scripts\python.exe" -u scripts\demo_server.py
pause
