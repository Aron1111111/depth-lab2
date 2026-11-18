@echo off
cd /d "%~dp0"
set "PYTHON=D:\Anaconda\Anaconda\envs\yolov8_env\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" sensor_fusion.py
pause
