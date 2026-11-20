@echo off
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  set PY=..\.venv\Scripts\python.exe
) else (
  set PY=python
)
%PY% A2_ransac_panorama.py
%PY% B2_kalman_fusion.py
%PY% C2_morphology.py
%PY% exam_ui.py
pause
