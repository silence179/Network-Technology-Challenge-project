@echo off
title Star CDN - Frontend Launcher

echo ========================================
echo   Star CDN Visual Frontend Launcher
echo ========================================
echo.

cd /d "%~dp0src\vis"

if not exist "node_modules\" (
    echo [INFO] Dependencies not found. Running npm install...
    echo.
    call npm install
    if errorlevel 1 (
        echo.
        echo [ERROR] npm install failed. Please check Node.js installation.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Dependencies installed.
    echo.
)

echo [START] Launching Vite dev server...
echo.
echo Browser will open automatically once ready.
echo Close this window or press Ctrl+C to stop.
echo.

call npx vite --open --host
