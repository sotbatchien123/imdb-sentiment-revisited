@echo off
cd /d "%~dp0.."
".venv-train\Scripts\python.exe" -u scripts\train_finetune_headtail.py --config configs\finetune_roberta_large_v1.json
pause
