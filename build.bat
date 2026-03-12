@echo off
cd /d "%~dp0"
title YupooScraper Build

echo ========================================
echo   YupooScraper Build
echo ========================================
echo.

where pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] PyInstaller not found, installing...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [X] PyInstaller install failed
        pause
        exit /b 1
    )
)

echo [1/3] Cleaning old build...
if exist build rmdir /s /q build
if exist dist\YupooScraper rmdir /s /q dist\YupooScraper

echo [2/3] Building...
echo.
pyinstaller yupoo_scraper.spec --noconfirm
if %errorlevel% neq 0 (
    echo.
    echo [X] Build failed!
    pause
    exit /b 1
)

echo [3/3] Creating data dirs...
if not exist dist\YupooScraper\data mkdir dist\YupooScraper\data
if not exist dist\YupooScraper\downloads mkdir dist\YupooScraper\downloads

echo.
echo ========================================
echo   Build done!
echo   Output: dist\YupooScraper\
echo   Run:    dist\YupooScraper\YupooScraper.exe
echo ========================================
pause
