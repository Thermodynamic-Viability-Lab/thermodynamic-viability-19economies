"""
Portable Windows CUDA Driver + NVRTC wrapper using ctypes (zero external PyTorch/CuPy compile toolkits).
- Uses nvcuda.dll from the NVIDIA display driver.
- Uses nvrtc64_*.dll from pip package `nvidia-cuda-nvrtc-cu12` (or CUDA_PATH / PATH).
- Compiles CUDA C++ source to forward-compatible `compute_80` PTX so the active GPU driver
  JIT-compiles natively to Ada Lovelace (`sm_89`, RTX 4080S) AND Blackwell (`sm_120`, RTX 5080).
"""
import ctypes
import glob
import os
import sys
import numpy as np

# CUDA Driver API / NVRTC constants
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def _find_nvrtc_dll():
    candidates = []
    # 1. pip package `nvidia-cuda-nvrtc-cu12` (namespace package -> use __path__)
    try:
        import nvidia.cuda_nvrtc as pkg
        for p in getattr(pkg, "__path__", []):
            bin_dir = os.path.join(p, "bin")
            candidates.extend(sorted(glob.glob(os.path.join(bin_dir, "nvrtc64_*.dll"))))
    except Exception:
        pass

    # 2. Active venv / site-packages fallback
    for root in sys.path:
        if not root:
            continue
        bin_dir = os.path.join(root, "nvidia", "cuda_nvrtc", "bin")
        if os.path.isdir(bin_dir):
            candidates.extend(sorted(glob.glob(os.path.join(bin_dir, "nvrtc64_*.dll"))))

    # 3. CUDA_PATH / PATH fallback
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.isdir(os.path.join(cuda_path, "bin")):
        candidates.extend(sorted(glob.glob(os.path.join(cuda_path, "bin", "nvrtc64_*.dll"))))

    # Prefer non-.alt primary DLLs
    primary = [c for c in candidates if ".alt." not in os.path.basename(c).lower() and os.path.isfile(c)]
    if primary:
        return primary[-1]
    existing = [c for c in candidates if os.path.isfile(c)]
    if existing:
        return existing[-1]
    raise RuntimeError(
        "Could not locate nvrtc64_*.dll. Please run 01_setup_windows_env.bat "
        "or `pip install nvidia-cuda-nvrtc-cu12`."
    )


class CudaError(RuntimeError):
    pass


class DeviceArray:
    """Managed GPU buffer with deterministic deallocation."""

    def __init__(self, driver, ptr_val, nbytes, shape=None, dtype=np.float64):
        self.driver = driver
        self.ptr = ctypes.c_uint64(ptr_val)
        self.nbytes = int(nbytes)
        self.shape = tuple(shape) if shape is not None else (nbytes,)
        self.dtype = np.dtype(dtype)
        self._freed = False

    def copy_from_host(self, arr):
        arr_c = np.ascontiguousarray(arr, dtype=self.dtype)
        if arr_c.nbytes != self.nbytes:
            raise ValueError(f"Host buffer {arr_c.nbytes}B != Device buffer {self.nbytes}B")
        self.driver.htod(self.ptr, arr_c)

    def to_host(self):
        out = np.empty(self.shape, dtype=self.dtype)
        self.driver.dtoh(out, self.ptr, self.nbytes)
        return out

    def zero(self):
        self.driver.memset_d8(self.ptr, 0, self.nbytes)

    def free(self):
        if not self._freed and self.ptr.value != 0:
            self.driver.mem_free(self.ptr)
            self.ptr = ctypes.c_uint64(0)
            self._freed = True

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CudaContext:
    """Thin, reliable ctypes interface to nvcuda.dll + nvrtc64_*.dll."""

    def __init__(self, device_id=0):
        # Load NVCUDA
        try:
            self.cu = ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            raise CudaError("Failed to load nvcuda.dll. Ensure an NVIDIA GPU + driver is installed.") from exc

        # Load NVRTC (+ add DLL directory to BOTH PATH and add_dll_directory so nvrtc-builtins64_*.dll resolves cleanly on Windows)
        self.nvrtc_dll_path = _find_nvrtc_dll()
        nvrtc_bin_dir = os.path.dirname(self.nvrtc_dll_path)
        os.environ["PATH"] = nvrtc_bin_dir + os.pathsep + os.environ.get("PATH", "")
        self._dll_dir_handle = None
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(nvrtc_bin_dir)
        # Preload builtins DLL if present in the same folder
        for b_dll in sorted(glob.glob(os.path.join(nvrtc_bin_dir, "nvrtc-builtins64_*.dll"))):
            try:
                ctypes.WinDLL(b_dll)
            except OSError:
                pass
        self.nvrtc = ctypes.WinDLL(self.nvrtc_dll_path)

        self._bind_signatures()
        self._check_cu(self.cu.cuInit(0), "cuInit")

        dev = ctypes.c_int()
        self._check_cu(self.cu.cuDeviceGet(ctypes.byref(dev), int(device_id)), "cuDeviceGet")
        self.device = dev.value

        name_buf = ctypes.create_string_buffer(256)
        self._check_cu(self.cu.cuDeviceGetName(name_buf, 256, self.device), "cuDeviceGetName")
        self.gpu_name = name_buf.value.decode("utf-8", errors="replace").strip()

        vram_bytes = ctypes.c_size_t()
        self._check_cu(self.cu.cuDeviceTotalMem_v2(ctypes.byref(vram_bytes), self.device), "cuDeviceTotalMem_v2")
        self.vram_gb = float(vram_bytes.value) / (1024.0 ** 3)

        maj, min_, mp = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._check_cu(self.cu.cuDeviceGetAttribute(ctypes.byref(maj), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device), "SM major")
        self._check_cu(self.cu.cuDeviceGetAttribute(ctypes.byref(min_), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device), "SM minor")
        self._check_cu(self.cu.cuDeviceGetAttribute(ctypes.byref(mp), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.device), "MP count")
        self.sm_major = int(maj.value)
        self.sm_minor = int(min_.value)
        self.sm_str = f"sm_{self.sm_major}{self.sm_minor}"
        self.mp_count = int(mp.value)

        drv_ver = ctypes.c_int()
        self._check_cu(self.cu.cuDriverGetVersion(ctypes.byref(drv_ver)), "cuDriverGetVersion")
        self.driver_cuda_version = f"{drv_ver.value // 1000}.{(drv_ver.value % 1000) // 10}"

        n_maj, n_min = ctypes.c_int(), ctypes.c_int()
        self._check_nvrtc(self.nvrtc.nvrtcVersion(ctypes.byref(n_maj), ctypes.byref(n_min)), "nvrtcVersion")
        self.nvrtc_version = f"{n_maj.value}.{n_min.value}"

        self.ctx = ctypes.c_void_p()
        self._check_cu(self.cu.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.device), "cuCtxCreate_v2")
        self._modules = []

    def _bind_signatures(self):
        cu = self.cu
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cu.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        cu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cu.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cu.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        cu.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        cu.cuCtxSynchronize.argtypes = []
        cu.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        cu.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
        cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]

        cu.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
            ctypes.c_uint, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)
        ]
        cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)
        ]

        nv = self.nvrtc
        nv.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        nv.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        nv.nvrtcGetErrorString.restype = ctypes.c_char_p
        nv.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)
        ]
        nv.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        nv.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        nv.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nv.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nv.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nv.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    def _check_cu(self, err, where=""):
        if err != 0:
            msg_ptr = ctypes.c_char_p()
            try:
                self.cu.cuGetErrorString(err, ctypes.byref(msg_ptr))
                err_str = msg_ptr.value.decode("utf-8", errors="replace") if msg_ptr.value else f"code {err}"
            except Exception:
                err_str = f"code {err}"
            raise CudaError(f"CUDA Driver error at {where}: {err_str} ({err})")

    def _check_nvrtc(self, err, where=""):
        if err != 0:
            s = self.nvrtc.nvrtcGetErrorString(err)
            err_str = s.decode("utf-8", errors="replace") if s else f"code {err}"
            raise CudaError(f"NVRTC error at {where}: {err_str} ({err})")

    def info_dict(self):
        return {
            "gpu_name": self.gpu_name,
            "vram_gb": round(self.vram_gb, 2),
            "compute_capability": self.sm_str,
            "multiprocessor_count": self.mp_count,
            "driver_cuda_version": self.driver_cuda_version,
            "nvrtc_version": self.nvrtc_version,
            "nvrtc_dll_path": self.nvrtc_dll_path,
        }

    def compile_and_load(self, cu_source, src_name="sfc_bp_open_economy.cu", ptx_arch="compute_80"):
        """
        Compiles CUDA C++ source to forward-compatible PTX (`compute_80`), then JIT-loads into
        the active GPU context (`sm_89`, `sm_120`, etc.).
        """
        prog = ctypes.c_void_p()
        src_bytes = cu_source.encode("utf-8") if isinstance(cu_source, str) else cu_source
        self._check_nvrtc(
            self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, src_name.encode("utf-8"), 0, None, None),
            "nvrtcCreateProgram"
        )
        compile_log = ""
        try:
            opts = [
                f"--gpu-architecture={ptx_arch}".encode("utf-8"),
                b"--std=c++14",
            ]
            opts_arr = (ctypes.c_char_p * len(opts))(*opts)
            res = self.nvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)

            log_sz = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            if log_sz.value > 1:
                buf = ctypes.create_string_buffer(log_sz.value)
                self.nvrtc.nvrtcGetProgramLog(prog, buf)
                compile_log = buf.value.decode("utf-8", errors="replace").strip()

            if res != 0:
                raise CudaError(f"NVRTC compilation failed ({res}):\n{compile_log}")

            ptx_sz = ctypes.c_size_t()
            self._check_nvrtc(self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz)), "nvrtcGetPTXSize")
            ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
            self._check_nvrtc(self.nvrtc.nvrtcGetPTX(prog, ptx_buf), "nvrtcGetPTX")
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        mod = ctypes.c_void_p()
        err_buf = ctypes.create_string_buffer(16384)
        jit_opts = (ctypes.c_int * 2)(CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        jit_vals = (ctypes.c_void_p * 2)(
            ctypes.cast(err_buf, ctypes.c_void_p),
            ctypes.c_void_p(16384)
        )
        res_mod = self.cu.cuModuleLoadDataEx(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p), 2, jit_opts, jit_vals)
        if res_mod != 0:
            jit_msg = err_buf.value.decode("utf-8", errors="replace")
            raise CudaError(f"cuModuleLoadDataEx failed ({res_mod}):\n{jit_msg}")

        self._modules.append(mod)
        return CudaModule(self, mod, compile_log)

    def alloc_array(self, arr_or_shape, dtype=np.float64):
        if isinstance(arr_or_shape, np.ndarray):
            arr_c = np.ascontiguousarray(arr_or_shape)
            ptr = ctypes.c_uint64()
            self._check_cu(self.cu.cuMemAlloc_v2(ctypes.byref(ptr), max(arr_c.nbytes, 8)), "cuMemAlloc_v2")
            dev = DeviceArray(self, ptr.value, arr_c.nbytes, shape=arr_c.shape, dtype=arr_c.dtype)
            if arr_c.nbytes > 0:
                dev.copy_from_host(arr_c)
            return dev
        else:
            shape = (arr_or_shape,) if isinstance(arr_or_shape, int) else tuple(arr_or_shape)
            dt = np.dtype(dtype)
            nbytes = int(np.prod(shape)) * dt.itemsize
            ptr = ctypes.c_uint64()
            self._check_cu(self.cu.cuMemAlloc_v2(ctypes.byref(ptr), max(nbytes, 8)), "cuMemAlloc_v2")
            dev = DeviceArray(self, ptr.value, nbytes, shape=shape, dtype=dt)
            if nbytes > 0:
                dev.zero()
            return dev

    def htod(self, ptr, arr_c):
        self._check_cu(
            self.cu.cuMemcpyHtoD_v2(ptr, arr_c.ctypes.data_as(ctypes.c_void_p), arr_c.nbytes),
            "cuMemcpyHtoD_v2"
        )

    def dtoh(self, dst_arr, ptr, nbytes):
        self._check_cu(
            self.cu.cuMemcpyDtoH_v2(dst_arr.ctypes.data_as(ctypes.c_void_p), ptr, nbytes),
            "cuMemcpyDtoH_v2"
        )

    def memset_d8(self, ptr, val, nbytes):
        self._check_cu(self.cu.cuMemsetD8_v2(ptr, int(val), int(nbytes)), "cuMemsetD8_v2")

    def mem_free(self, ptr):
        self._check_cu(self.cu.cuMemFree_v2(ptr), "cuMemFree_v2")

    def sync(self):
        self._check_cu(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self):
        for m in self._modules:
            try:
                self.cu.cuModuleUnload(m)
            except Exception:
                pass
        self._modules.clear()
        if getattr(self, "ctx", None):
            try:
                self.cu.cuCtxDestroy_v2(self.ctx)
            except Exception:
                pass
            self.ctx = None


class CudaModule:
    def __init__(self, driver, mod_handle, compile_log=""):
        self.driver = driver
        self.mod = mod_handle
        self.compile_log = compile_log
        self._funcs = {}

    def get_function(self, name):
        if name not in self._funcs:
            fn = ctypes.c_void_p()
            self.driver._check_cu(
                self.driver.cu.cuModuleGetFunction(ctypes.byref(fn), self.mod, name.encode("utf-8")),
                f"cuModuleGetFunction({name})"
            )
            self._funcs[name] = fn
        return self._funcs[name]

    def launch(self, func_name, grid_x, block_x, args, grid_y=1, grid_z=1, sync=True):
        fn = self.get_function(func_name)
        packed = []
        ptrs = []
        for a in args:
            if isinstance(a, DeviceArray):
                cval = ctypes.c_uint64(a.ptr.value)
            elif isinstance(a, np.floating) or isinstance(a, float):
                cval = ctypes.c_double(float(a))
            elif isinstance(a, (np.integer, int)):
                cval = ctypes.c_int32(int(a))
            elif isinstance(a, ctypes.c_void_p) or isinstance(a, ctypes.c_uint64) or isinstance(a, ctypes.c_int32) or isinstance(a, ctypes.c_uint32) or isinstance(a, ctypes.c_double):
                cval = a
            else:
                raise TypeError(f"Unsupported kernel argument type: {type(a)}")
            packed.append(cval)
            ptrs.append(ctypes.cast(ctypes.byref(cval), ctypes.c_void_p))

        param_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        self.driver._check_cu(
            self.driver.cu.cuLaunchKernel(
                fn,
                int(grid_x), int(grid_y), int(grid_z),
                int(block_x), 1, 1,
                0, None,
                param_arr, None
            ),
            f"cuLaunchKernel({func_name})"
        )
        if sync:
            self.driver.sync()
