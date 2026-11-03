@echo off
cd /d "%~dp0.."
echo pause>work_dirs\imdb_full_v1\PAUSE
echo Pause requested. Wait for status_live.json to show paused after saving the epoch.
pause
