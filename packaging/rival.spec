# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH).resolve().parent
bot_root = project_root / "bot"

datas = [
    (str(bot_root / "models"), "models"),
    (str(bot_root / "collision_meshes"), "collision_meshes"),
    (str(project_root / "third_party" / "wisp"), "third_party/wisp"),
    (str(project_root / "packaging" / "RELEASE_README.md"), "."),
]

analysis = Analysis(
    [str(bot_root / "bot.py")],
    pathex=[str(bot_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "ruff"],
    noarchive=False,
    optimize=0,
)

python_modules = PYZ(analysis.pure)

executable = EXE(
    python_modules,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="RivalDev",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RivalDev",
)
