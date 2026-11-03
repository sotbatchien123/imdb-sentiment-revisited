@echo off
cd /d "%~dp0.."
".venv-train\Scripts\python.exe" -u scripts\train_finetune_headtail.py
pause
