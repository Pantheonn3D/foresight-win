# PyInstaller spec: one-folder windowed build at dist/Foresight/Foresight.exe
# Build with:  uv run pyinstaller packaging/foresight.spec --noconfirm
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parent

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("websockets")
    + collect_submodules("foresight")
    + ["pyaudiowpatch", "soxr", "multipart", "python_multipart"]
)

a = Analysis(
    [str(root / "packaging" / "launch.py")],
    pathex=[str(root)],
    datas=[(str(root / "foresight" / "web"), "foresight/web")] + collect_data_files("webview"),
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "PIL", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Foresight",
    console=False,
    icon=str(root / "packaging" / "foresight.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="Foresight")
