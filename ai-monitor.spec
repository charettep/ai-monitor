# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ai-monitor standalone binary."""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# Bundle PyQt6 platform plugins and pyqtgraph
for pkg in ("PyQt6", "pyqtgraph"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["ai_monitor.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ai-monitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
)
