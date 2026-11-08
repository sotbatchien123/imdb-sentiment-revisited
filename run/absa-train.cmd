@echo off
rem Huan luyen 3 mo hinh ABSA x 3 seed. Moi seed mot tien trinh rieng de VRAM duoc giai phong het
rem (chay chung mot tien trinh lam RoBERTa-large tran VRAM o seed thu hai). Seed da xong se duoc bo qua.
rem Tao file work_dirs\<version>\PAUSE de tam dung o cuoi epoch.
cd /d "%~dp0.."
for %%C in (absa_bilstm_v1 absa_bert_base_v1 absa_roberta_large_v1) do (
  for %%S in (42 43 44) do (
    ".venv-train\Scripts\python.exe" -u scripts\train_absa.py --config configs\%%C.json --seed %%S || goto :eof
  )
)
pause
