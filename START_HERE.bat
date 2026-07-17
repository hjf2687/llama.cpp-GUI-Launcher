@echo off
title llama.cpp GUI Launcher
echo.
echo ========================================
echo   llama.cpp GUI Launcher (Optimized)
echo ========================================
echo.

cd /d "%~dp0"

:: Check Python (try "py", then "python")
where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python"
    ) else (
        echo [ERROR] Python not found. Please install Python and add it to PATH.
        echo.
        pause
        exit /b 1
    )
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

%PYTHON_CMD% llama_launcher.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Launcher exited with error code: %errorlevel%
    echo Check the log for details.
)

echo.
echo Launcher closed.
pause
