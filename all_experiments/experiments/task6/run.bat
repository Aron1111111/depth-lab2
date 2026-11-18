@echo off
setlocal
set "LAB_PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%LAB_PY%" set "LAB_PY=D:\Anaconda\Anaconda\python.exe"
if not exist "%LAB_PY%" set "LAB_PY=python"
"%LAB_PY%" "%~dp0classification.py"
if errorlevel 1 pause
