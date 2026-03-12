# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — yupoo_scraper (onedir 模式)"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# ── 隐式导入 ──────────────────────────────────────────────────
hidden = [
    # lxml C 扩展
    "lxml._elementpath",
    "lxml.etree",
    # open_clip 内部注册
    *collect_submodules("open_clip"),
    # rembg 模型 session
    *collect_submodules("rembg"),
    # faiss
    "faiss",
    "faiss.swigfaiss",
    # sklearn 子模块（聚类）
    "sklearn.cluster",
    "sklearn.cluster._agglomerative",
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    # torch（torchvision 项目未使用，不打包）
    "torch",
    # qasync
    "qasync",
    # cv2
    "cv2",
    # PIL
    "PIL",
    # playwright — 打包后通过系统 Chrome 或手动安装 chromium
    "playwright",
    "playwright.async_api",
]

# ── 数据文件（模型权重、配置等）──────────────────────────────
datas = []
datas += collect_data_files("open_clip")
datas += collect_data_files("rembg")
datas += collect_data_files("torch")       # schema 等元数据

# ── 动态库（torch DLL 互相依赖，必须全部收集）────────────────
binaries_extra = []
binaries_extra += collect_dynamic_libs("torch")

# ── 分析 ──────────────────────────────────────────────────────
a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries_extra,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rthook_torch.py"],
    excludes=[
        # 不需要的 Qt 模块（省空间）
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtWebSockets",
        "PySide6.QtPdf",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        # tkinter（不需要）
        "tkinter",
        "_tkinter",
        # test 模块
        "pytest",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir 模式
    name="YupooScraper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX 压缩对 torch 等大库容易出错，关掉
    console=False,  # GUI 应用，不弹控制台
    icon=None,  # 如有 .ico 图标文件可在此指定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="YupooScraper",
)
