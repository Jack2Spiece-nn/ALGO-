@echo off
echo GARCH Trading Strategy - Windows Setup
echo ======================================
echo.

cd /d "%~dp0"

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Trying user installation instead...
        goto :user_install
    )
)

echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    goto :user_install
)

echo [4/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo SUCCESS: Setup complete!
echo.
echo To run the trading system:
echo   1. Double-click start_trading.bat
echo   2. Or run: venv\Scripts\activate.bat then python src\main.py
echo.
goto :end

:user_install
echo.
echo Installing to user directory (no virtual environment)...
pip install --user -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies to user directory
    pause
    exit /b 1
)

echo.
echo SUCCESS: Setup complete (user installation)!
echo.
echo To run the trading system:
echo   python src\main.py
echo.

:end
pause