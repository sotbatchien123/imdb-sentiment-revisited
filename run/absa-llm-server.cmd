@echo off
rem Mo llama-server (Qwen3-8B Q4_K_M) tren GPU cho buoc gan nhan ABSA. Can GPU ranh (~6,5 GB VRAM).
rem Doi duong dan neu de llama.cpp / model o noi khac.
set LLAMA_DIR=E:\llm
"%LLAMA_DIR%\bin\llama-server.exe" -m "%LLAMA_DIR%\models\Qwen3-8B-Q4_K_M.gguf" -ngl 99 -c 12288 -np 2 -fa on -ctk q8_0 -ctv q8_0 --jinja --host 127.0.0.1 --port 8080
pause
