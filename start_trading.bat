@echo off
echo GARCH Trading Strategy for Arjay Siega
echo ======================================

cd /d "%~dp0"

rem Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Running startup test first...
python test_windows_startup.py

if errorlevel 1 (
    echo.
    echo ERROR: Startup tests failed!
    echo Please run setup_windows.bat first to install dependencies
    pause
    exit /b 1
)

echo.
echo Tests passed! Starting trading engine...
echo.
python src\main.py

pause