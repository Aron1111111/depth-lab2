@echo off
cd /d "%~dp0"
set "PYTHON=D:\002_codex\image_class\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=D:\Anaconda\Anaconda\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" calibration_lab.py
pause
