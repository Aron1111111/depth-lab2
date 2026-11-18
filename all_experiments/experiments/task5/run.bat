@echo off
setlocal
set "LAB_PY=D:\Anaconda\Anaconda\envs\yolov8_env\python.exe"
if not exist "%LAB_PY%" set "LAB_PY=python"
"%LAB_PY%" "%~dp0panorama.py"
if errorlevel 1 pause
