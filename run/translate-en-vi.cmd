@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv-translation\Scripts\python.exe" (
echo Run run\setup-translation.cmd first.
pause
exit /b 1
)
if not exist "data\translated" mkdir "data\translated"
".venv-translation\Scripts\python.exe" scripts\translate_en_vi.py --input data\raw\review_en.txt --output data\translated\review_vi.txt
pause
