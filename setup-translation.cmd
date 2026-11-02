@echo off
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
set "TEMP=%CD%	mp"
set "TMP=%CD%	mp"
if not exist ".venv-translation\Scripts\python.exe" python -m venv .venv-translation
if errorlevel 1 goto fail
".venv-translation\Scripts\python.exe" -m pip install -r requirements-translation.txt
if errorlevel 1 goto fail
echo Setup completed.
pause
exit /b 0
:fail
echo Setup failed. Read the error above.
pause
exit /b 1
