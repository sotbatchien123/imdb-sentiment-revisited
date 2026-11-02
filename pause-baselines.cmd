@echo off
cd /d "%~dp0"
echo pause>work_dirs\horror_frozen_bert_v1\PAUSE
echo Pause requested. The current epoch will finish and save its checkpoint.
echo Wait until status.json shows paused before shutting down.
pause
