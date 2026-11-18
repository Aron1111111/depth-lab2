@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe lab2\depth_lab_ui.py
) else (
  python lab2\depth_lab_ui.py
)
