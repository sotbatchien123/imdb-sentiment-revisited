@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-translation\Scripts\python.exe" (
echo Run setup-translation.cmd first.
pause
exit /b 1
)
".venv-translation\Scripts\python.exe" scripts	ranslate_en_vi.py --input dataaweview_en.txt --output data	ranslatedeview_vi.txt
pause
