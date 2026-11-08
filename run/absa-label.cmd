@echo off
rem Gan nhan silver (prompt v3 co bang chung) cho ca 3 split; chay lai se tiep tuc tu cho dang do. Mo absa-llm-server.cmd truoc.
cd /d "%~dp0.."
".venv-train\Scripts\python.exe" -u scripts\absa_label_llm.py --config configs\absa_labeling_v2.json --splits validation train test
pause
