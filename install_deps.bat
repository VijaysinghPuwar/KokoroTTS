@echo off
setlocal
where python >nul 2>nul
if %errorlevel% neq 0 (
  echo Python not found in PATH. Please install Python 3.12+ and re-run.
  pause & exit /b 1
)
python -m venv .venv
if %errorlevel% neq 0 (
  echo Failed to create virtual environment.
  pause & exit /b 1
)
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
echo.
echo Dependencies installed. Now run: run_app.bat
pause
