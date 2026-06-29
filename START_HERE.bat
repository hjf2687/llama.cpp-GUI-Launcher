@echo off
title llama.cpp GUI Launcher
echo.
echo ========================================
echo   llama.cpp GUI Launcher (Optimized)
echo ========================================
echo.

cd /d "%~dp0"

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    echo.
    pause
    exit /b 1
)

:: Check launcher script
if not exist "llama_launcher.py" (
    echo [ERROR] llama_launcher.py not found in current directory.
    echo.
    pause
    exit /b 1
)

echo Starting launcher...
echo.

python llama_launcher.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Launcher exited with error code: %errorlevel%
    echo Check the log for details.
)

echo.
echo Launcher closed.
pause
