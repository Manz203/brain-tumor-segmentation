@echo off
setlocal
cd /d "%~dp0"
if not exist "neuroseg_results\best_model.pth" (
  echo Trained checkpoint missing. Extract the complete trained-project artifact, not just the source repository.
  pause
  exit /b 1
)
where python >nul 2>nul || (
  echo Python 3.11 or newer is required.
  pause
  exit /b 1
)
if not exist ".neuroseg-env\Scripts\python.exe" (
  python -m venv .neuroseg-env || goto fail
)
set "PY=.neuroseg-env\Scripts\python.exe"
if not exist ".neuroseg-env\installed.ok" (
  "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto fail
  "%PY%" -m pip install -r requirements-v2.txt || goto fail
  echo ready>".neuroseg-env\installed.ok"
)
set "NEUROSEG_RESULTS=%CD%\neuroseg_results"
set "NEUROSEG_LOAD_CANDIDATES=1"
echo Starting local research application on http://127.0.0.1:8000
"%PY%" launch_neuroseg.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo NeuroSeg could not start. Read the error above.
pause
exit /b 1
