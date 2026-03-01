@echo off
chcp 65001 >nul
title 产品上架工具
cd /d "%~dp0"
python -m yupoo_scraper
if errorlevel 1 (
    echo.
    echo 启动失败，请检查 Python 环境和依赖是否已安装。
    echo 安装依赖: pip install -r requirements.txt
    pause
)
