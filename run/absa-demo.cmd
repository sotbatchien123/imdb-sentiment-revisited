@echo off
rem Demo ABSA: mo http://127.0.0.1:8766 sau khi thay dong "San sang". Them --device cuda neu GPU dang ranh.
cd /d "%~dp0.."
".venv-train\Scripts\python.exe" -u scripts\absa_demo_server.py %*
pause
