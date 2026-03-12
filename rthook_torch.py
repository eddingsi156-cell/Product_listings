"""Runtime hook — pre-load torch DLLs before torch.__init__ runs.

PyInstaller 打包后，多个包（faiss、sklearn、torch）各自携带 MSVC/OpenMP
运行时 DLL 的不同版本。torch 自己的 _load_dll_libraries() 使用
LoadLibraryExW + LOAD_LIBRARY_SEARCH_DEFAULT_DIRS (0x1100) 限制了搜索范围，
在打包环境下容易加载到错误版本的依赖，导致 c10.dll DllMain 失败 (WinError 1114)。

解决方案：在 torch 被 import 之前，用标准 ctypes.CDLL (LoadLibraryW) 按正确
顺序预加载全部 torch DLL。后续 torch 的 LoadLibraryExW 调用会直接获得已加载
的句柄，不再触发 DllMain。
"""
import ctypes
import glob
import os
import sys

_torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
if os.path.isdir(_torch_lib):
    # 1. 注册 DLL 搜索目录
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
        os.add_dll_directory(sys._MEIPASS)
    os.environ["PATH"] = (
        _torch_lib + ";" + sys._MEIPASS + ";" + os.environ.get("PATH", "")
    )

    # 2. 确保 MSVC 运行时从 _MEIPASS 根目录加载（避免拿到其他包的版本）
    for _crt in ("vcruntime140.dll", "msvcp140.dll", "vcruntime140_1.dll"):
        _crt_path = os.path.join(sys._MEIPASS, _crt)
        if os.path.exists(_crt_path):
            try:
                ctypes.CDLL(_crt_path)
            except OSError:
                pass

    # 3. 按依赖顺序预加载 torch DLL（关键：torch_global_deps 必须在 c10 之前）
    _priority = [
        "torch_global_deps.dll",
        "libiomp5md.dll",
        "libiompstubs5md.dll",
        "uv.dll",
        "c10.dll",
        "torch_cpu.dll",
        "torch.dll",
        "shm.dll",
        "torch_python.dll",
    ]
    _loaded = set()
    for _name in _priority:
        _path = os.path.join(_torch_lib, _name)
        if os.path.exists(_path):
            try:
                ctypes.CDLL(_path)
                _loaded.add(_name.lower())
            except OSError:
                pass

    # 4. 兜底：加载上面 priority 列表遗漏的 DLL
    for _path in sorted(glob.glob(os.path.join(_torch_lib, "*.dll"))):
        if os.path.basename(_path).lower() not in _loaded:
            try:
                ctypes.CDLL(_path)
            except OSError:
                pass
