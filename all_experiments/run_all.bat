@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe all_experiments\combined_ui.py
) else (
  python all_experiments\combined_ui.py
)
