"""Minimal ctypes binding of the RKNN C API (librknnrt) for zero-copy, native-layout I/O.

Struct layouts and constants follow Rockchip's rknn_api.h (airockchip/rknn-toolkit2,
runtime 2.5.0). Every call releases the GIL (ctypes), so lanes run in parallel threads.

    m = Model(path, core=0)               # rknn_init + rknn_set_core_mask
    m2 = m.dup(core=1)                    # rknn_dup_context: shares the weights
    m.bind(native=True)                   # rknn_create_mem + rknn_set_io_mem for every input/output
    m.inp[i]  -> numpy view of input i's DMA buffer, in its native layout (fp16)
    m.out[i]  -> numpy view of output i's DMA buffer
    m.run()                               # rknn_run (releases the GIL: ctypes)
"""
import ctypes as C
import os

import numpy as np

lib = C.CDLL(os.environ.get("RKNN_LIB", "librknnrt.so"))
ctx_t = C.c_uint64
MAX_DIMS, MAX_NAME = 16, 256
QUERY_IN_OUT_NUM, QUERY_INPUT_ATTR, QUERY_OUTPUT_ATTR, QUERY_PERF_RUN, QUERY_MEM_SIZE = 0, 1, 2, 4, 6
QUERY_NATIVE_INPUT_ATTR, QUERY_NATIVE_OUTPUT_ATTR = 8, 9
QUERY_NATIVE_NHWC_INPUT_ATTR, QUERY_NATIVE_NHWC_OUTPUT_ATTR = 10, 11
FMT = {0: "NCHW", 1: "NHWC", 2: "NC1HWC2", 3: "UNDEFINED"}
TYPE = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8, 4: np.int16, 5: np.uint16, 6: np.int32,
        7: np.uint32, 8: np.int64, 9: np.bool_}
CORE = {None: 0, "auto": 0, 0: 1, 1: 2, 2: 4, "0": 1, "1": 2, "2": 4, "0_1": 3, "0_1_2": 7}
FLAG_SHARE_WEIGHT_MEM = 0x20
FLAG_INTERNAL_ALLOC_OUTSIDE = 0x200
SYNC_TO, SYNC_FROM = 1, 2


class InOutNum(C.Structure):
    _fields_ = [("n_input", C.c_uint32), ("n_output", C.c_uint32)]


class TensorAttr(C.Structure):
    _fields_ = [("index", C.c_uint32), ("n_dims", C.c_uint32), ("dims", C.c_uint32 * MAX_DIMS),
                ("name", C.c_char * MAX_NAME), ("n_elems", C.c_uint32), ("size", C.c_uint32),
                ("fmt", C.c_int), ("type", C.c_int), ("qnt_type", C.c_int), ("fl", C.c_int8),
                ("zp", C.c_int32), ("scale", C.c_float), ("w_stride", C.c_uint32),
                ("size_with_stride", C.c_uint32), ("pass_through", C.c_uint8), ("h_stride", C.c_uint32)]

    def shape(self):
        return tuple(self.dims[i] for i in range(self.n_dims))

    def __repr__(self):
        return (f"<{self.name.decode()} {self.shape()} {FMT.get(self.fmt)} {TYPE.get(self.type).__name__} "
                f"size {self.size} w_stride {self.w_stride} sws {self.size_with_stride}>")


class TensorMem(C.Structure):
    _fields_ = [("virt_addr", C.c_void_p), ("phys_addr", C.c_uint64), ("fd", C.c_int32), ("offset", C.c_int32),
                ("size", C.c_uint32), ("flags", C.c_uint32), ("priv_data", C.c_void_p)]


class MemSize(C.Structure):
    _fields_ = [("total_weight_size", C.c_uint32), ("total_internal_size", C.c_uint32),
                ("total_dma_allocated_size", C.c_uint64), ("total_sram_size", C.c_uint32),
                ("free_sram_size", C.c_uint32), ("reserved", C.c_uint32 * 10)]


class PerfRun(C.Structure):
    _fields_ = [("run_duration", C.c_int64)]


lib.rknn_init.argtypes = [C.POINTER(ctx_t), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
lib.rknn_dup_context.argtypes = [C.POINTER(ctx_t), C.POINTER(ctx_t)]
lib.rknn_destroy.argtypes = [ctx_t]
lib.rknn_query.argtypes = [ctx_t, C.c_int, C.c_void_p, C.c_uint32]
lib.rknn_set_core_mask.argtypes = [ctx_t, C.c_int]
lib.rknn_run.argtypes = [ctx_t, C.c_void_p]
lib.rknn_create_mem.argtypes = [ctx_t, C.c_uint32]
lib.rknn_create_mem.restype = C.POINTER(TensorMem)
lib.rknn_create_mem2.argtypes = [ctx_t, C.c_uint64, C.c_uint64]
lib.rknn_create_mem2.restype = C.POINTER(TensorMem)
lib.rknn_destroy_mem.argtypes = [ctx_t, C.POINTER(TensorMem)]
lib.rknn_set_io_mem.argtypes = [ctx_t, C.POINTER(TensorMem), C.POINTER(TensorAttr)]
lib.rknn_set_internal_mem.argtypes = [ctx_t, C.POINTER(TensorMem)]
lib.rknn_mem_sync.argtypes = [ctx_t, C.POINTER(TensorMem), C.c_int]


def _ok(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: {rc}")


def mem_view(mem, dtype, shape):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    assert n <= mem.contents.size, (n, mem.contents.size)
    buf = (C.c_char * n).from_address(mem.contents.virt_addr)
    return np.frombuffer(buf, dtype=dtype).reshape(shape)


class Model:
    def __init__(self, path=None, core=None, flags=0, _ctx=None):
        self.ctx = ctx_t(0)
        if _ctx is None:
            # from a buffer, not a path: rknn_dup_context segfaults on path-initialised contexts (librknnrt 2.5.0)
            data = open(path, "rb").read()
            self._buf = C.create_string_buffer(data, len(data))
            _ok(lib.rknn_init(C.byref(self.ctx), self._buf, len(data), flags, None), f"rknn_init {path}")
        else:
            self.ctx = _ctx
        self.path, self.mems = path, []
        if core is not None:
            _ok(lib.rknn_set_core_mask(self.ctx, CORE[core]), "rknn_set_core_mask")
        self.refresh()

    def refresh(self):
        n = InOutNum()
        _ok(lib.rknn_query(self.ctx, QUERY_IN_OUT_NUM, C.byref(n), C.sizeof(n)), "query io num")
        self.n_in, self.n_out = n.n_input, n.n_output
        self.in_attr = [self._attr(QUERY_INPUT_ATTR, i) for i in range(self.n_in)]
        self.out_attr = [self._attr(QUERY_OUTPUT_ATTR, i) for i in range(self.n_out)]
        self.in_native = [self._attr(QUERY_NATIVE_INPUT_ATTR, i) for i in range(self.n_in)]
        self.out_native = [self._attr(QUERY_NATIVE_OUTPUT_ATTR, i) for i in range(self.n_out)]
        self.names = [a.name.decode() for a in self.in_attr]

    def dup(self, core=None):
        new = ctx_t(0)
        _ok(lib.rknn_dup_context(C.byref(self.ctx), C.byref(new)), "rknn_dup_context")
        return Model(self.path, core, _ctx=new)

    def _attr(self, cmd, i):
        a = TensorAttr(); a.index = i
        _ok(lib.rknn_query(self.ctx, cmd, C.byref(a), C.sizeof(a)), f"query {cmd} {i}")
        return a

    def mem_size(self):
        m = MemSize()
        _ok(lib.rknn_query(self.ctx, QUERY_MEM_SIZE, C.byref(m), C.sizeof(m)), "query mem size")
        return dict(weight=m.total_weight_size, internal=m.total_internal_size, dma=m.total_dma_allocated_size,
                    sram=m.total_sram_size, sram_free=m.free_sram_size)

    def create_mem(self, size):
        # (rknn_create_mem2 with RKNN_FLAG_MEMORY_NON_CACHEABLE gives buffers numpy dies on: SIGBUS)
        m = lib.rknn_create_mem(self.ctx, size)
        if not m:
            raise RuntimeError("rknn_create_mem")
        self.mems.append(m)
        return m

    def set_internal(self, mem):
        _ok(lib.rknn_set_internal_mem(self.ctx, mem), "rknn_set_internal_mem")

    def io_sizes(self, native=True):
        """{('in'|'out', name): bytes} of the buffers bind() needs."""
        r = {}
        for kind, attrs in (("in", self.in_native if native else self.in_attr), ("out", self.out_native if native else self.out_attr)):
            plain = self.in_attr if kind == "in" else self.out_attr
            for a, b in zip(attrs, plain):
                r[kind, a.name.decode()] = max(a.size_with_stride, a.size, b.size, b.size_with_stride)
        return r

    def bind(self, native=True, shared=None):
        """One DMA buffer per input/output, in the native layout (pass-through, fp16).
        shared: {('in'|'out', name): mem} buffers to use instead of new ones (big enough)."""
        shared = shared or {}
        self.in_mem, self.out_mem, self.inp, self.out = [], [], [], []
        for kind, attrs in (("in", self.in_native if native else self.in_attr), ("out", self.out_native if native else self.out_attr)):
            for i, a in enumerate(attrs):
                a = TensorAttr.from_buffer_copy(a)
                if kind == "in":
                    a.pass_through = 1 if native else 0
                size = max(a.size_with_stride, a.size, self.out_attr[i].size if kind == "out" else 0)
                mem = shared.get((kind, a.name.decode())) or self.create_mem(size)
                _ok(lib.rknn_set_io_mem(self.ctx, mem, C.byref(a)), f"set_io_mem {kind}{i} {a}")
                v = mem_view(mem, TYPE[a.type], a.shape())    # the decoder's widths need no stride padding
                (self.in_mem if kind == "in" else self.out_mem).append(mem)
                (self.inp if kind == "in" else self.out).append(v)
        self.bound_in = [TensorAttr.from_buffer_copy(a) for a in (self.in_native if native else self.in_attr)]
        self.bound_out = [TensorAttr.from_buffer_copy(a) for a in (self.out_native if native else self.out_attr)]
        return self

    def sync_out(self):
        for m in self.out_mem:
            lib.rknn_mem_sync(self.ctx, m, SYNC_FROM)

    def run(self):
        _ok(lib.rknn_run(self.ctx, None), "rknn_run")

    def perf_run_us(self):
        p = PerfRun()
        lib.rknn_query(self.ctx, QUERY_PERF_RUN, C.byref(p), C.sizeof(p))
        return p.run_duration

    def release(self):
        for m in self.mems:
            lib.rknn_destroy_mem(self.ctx, m)
        self.mems = []
        lib.rknn_destroy(self.ctx)
