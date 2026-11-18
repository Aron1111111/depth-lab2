@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe lab2\restore_depth.py
) else (
  python lab2\restore_depth.py
)
pause
