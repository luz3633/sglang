"""
Unit tests for sglang.srt.hardware_backend.npu.utils.
"""

import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Import shims: sglang/__init__ pulls in the Lang API / hf_transformers_patches,
# which drag in optional deps (IPython, aiohttp, triton, ...). Keep them out so
# this UT runs on a CPU-only box with minimal mocking. These must be set up
# before any `import sglang.*` (e.g. register_npu_ci) triggers sglang/__init__.
for _mod in (
    "sglang.srt.utils.hf_transformers_patches",
    "sglang.lang.api",
    "sglang.lang.backend.runtime_endpoint",
    "sglang.lang.choices",
    "sglang.lang.global_config",
    "sglang.utils",
):
    sys.modules.setdefault(_mod, MagicMock())

_version = types.ModuleType("sglang.version")
_version.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _version

try:
    import triton  # noqa: F401
except ImportError:
    _triton = types.ModuleType("triton")
    _triton.jit = MagicMock(return_value=lambda f: f)
    _triton.autotune = lambda *a, **kw: (lambda f: f)
    sys.modules["triton"] = _triton
    sys.modules.setdefault("triton.language", MagicMock())
    sys.modules.setdefault("triton.backends", MagicMock())

# torch_npu is imported lazily inside init_npu_backend; register a stub so the
# import statement resolves when that function is exercised.
sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu import utils as npu_utils  # noqa: E402
from sglang.srt.hardware_backend.npu.utils import (  # noqa: E402
    NPUACLFormat,
    _call_once,
    _is_nz_aligned,
    get_indexer_weight_stream,
    get_routed_stream,
    get_share_stream,
    init_npu_backend,
    init_zbal,
    lazy_init_zbal_gva_mem,
    npu_format_cast,
    process_routed_expert,
    process_shared_expert,
    set_default_server_args,
    set_routed_stream,
    set_share_stream,
    wait_routed_stream,
    wait_share_stream,
)


class _FakeStream:
    """Stands in for an NPU stream; records wait_stream targets."""

    def __init__(self, name):
        self.name = name
        self.waited_on = []

    def wait_stream(self, other):
        self.waited_on.append(other)


class _FakeNPU:
    """Minimal torch.npu stand-in: stream creation + current-stream context."""

    def __init__(self):
        self.current = _FakeStream("default")
        self.created = []

    def Stream(self):
        stream = _FakeStream("npu")
        self.created.append(stream)
        return stream

    def current_stream(self):
        return self.current

    def stream(self, target):
        return _StreamContext(self, target)


class _StreamContext:
    def __init__(self, npu, target):
        self._npu = npu
        self._target = target

    def __enter__(self):
        self._prev = self._npu.current
        self._npu.current = self._target

    def __exit__(self, *exc):
        self._npu.current = self._prev


class _FakeTensor:
    """Minimal tensor stand-in exposing shape/dtype/device for npu_format_cast."""

    def __init__(self, shape, dtype=torch.float16, device="npu"):
        self.shape = shape
        self.dtype = dtype
        # torch on a CPU-only box rejects "npu" device strings, so use a fake
        # device that compares unequal to cpu and reports a non-meta type.
        self.device = SimpleNamespace(type=device)

    def dim(self):
        return len(self.shape)


class _NPUModuleTestBase(unittest.TestCase):
    """Restores torch.npu and torch.get_device_module, plus module-level state."""

    def _save_npu(self):
        self._prev_npu = getattr(torch, "npu", None)
        self._prev_gdm = getattr(torch, "get_device_module", None)
        self._fake = _FakeNPU()
        torch.npu = self._fake
        # utils.py routes stream work through torch.get_device_module(), so the
        # fake NPU must be returned on CPU-only boxes.
        torch.get_device_module = lambda: self._fake

    def _restore_npu(self):
        if self._prev_npu is None:
            del torch.npu
        else:
            torch.npu = self._prev_npu
        if self._prev_gdm is None:
            del torch.get_device_module
        else:
            torch.get_device_module = self._prev_gdm


class TestNPUACLFormat(unittest.TestCase):
    def test_values(self):
        self.assertEqual(NPUACLFormat.ACL_FORMAT_UNDEFINED, -1)
        self.assertEqual(NPUACLFormat.ACL_FORMAT_ND, 2)
        self.assertEqual(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ, 29)


class TestCallOnce(unittest.TestCase):
    def test_calls_fn_only_once(self):
        calls = []

        @_call_once
        def fn():
            calls.append(1)

        fn()
        fn()
        self.assertEqual(len(calls), 1)

    def test_preserves_metadata(self):
        @_call_once
        def my_func():
            pass

        self.assertEqual(my_func.__name__, "my_func")


class TestSetDefaultServerArgs(unittest.TestCase):
    def _call(self, npu_mem, **kwargs):
        args = SimpleNamespace(
            attention_backend="",
            prefill_attention_backend="",
            decode_attention_backend="",
            page_size=kwargs.pop("page_size", None),
            tp_size=kwargs.pop("tp_size", 1),
            chunked_prefill_size=kwargs.pop("chunked_prefill_size", None),
            disable_custom_all_reduce=False,
            enable_hierarchical_cache=kwargs.pop("enable_hierarchical_cache", False),
            hicache_io_backend=None,
            hicache_mem_layout=None,
        )
        args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(max_bs=kwargs.pop("max_bs", None))
        )
        args.use_mla_backend = lambda: kwargs.pop("use_mla", False)
        with patch.object(npu_utils, "get_npu_memory_capacity", return_value=npu_mem):
            set_default_server_args(args)
        return args

    def test_sets_ascend_backends(self):
        args = self._call(32 * 1024)
        self.assertEqual(args.attention_backend, "ascend")
        self.assertEqual(args.prefill_attention_backend, "ascend")
        self.assertEqual(args.decode_attention_backend, "ascend")

    def test_default_page_size(self):
        self.assertEqual(self._call(32 * 1024).page_size, 128)

    def test_keeps_user_page_size(self):
        self.assertEqual(self._call(32 * 1024, page_size=64).page_size, 64)

    def test_low_mem_small_tp(self):
        args = self._call(32 * 1024, tp_size=2)
        self.assertEqual(args.chunked_prefill_size, 4 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 16)

    def test_low_mem_large_tp(self):
        args = self._call(32 * 1024, tp_size=4)
        self.assertEqual(args.chunked_prefill_size, 4 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)

    def test_med_mem_small_tp(self):
        args = self._call(64 * 1024, tp_size=1)
        self.assertEqual(args.chunked_prefill_size, 8 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)

    def test_med_mem_large_tp(self):
        args = self._call(64 * 1024, tp_size=8)
        self.assertEqual(args.chunked_prefill_size, 8 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 256)

    def test_high_mem_leaves_defaults(self):
        args = self._call(100 * 1024)
        self.assertIsNone(args.chunked_prefill_size)
        self.assertIsNone(args.cuda_graph_config.decode.max_bs)

    def test_keeps_user_chunked_prefill_size(self):
        args = self._call(32 * 1024, chunked_prefill_size=4096)
        self.assertEqual(args.chunked_prefill_size, 4096)

    def test_disable_custom_all_reduce(self):
        self.assertTrue(self._call(32 * 1024).disable_custom_all_reduce)

    def test_hierarchical_cache_mla(self):
        args = self._call(32 * 1024, enable_hierarchical_cache=True, use_mla=True)
        self.assertEqual(args.hicache_io_backend, "kernel_ascend")
        self.assertEqual(args.hicache_mem_layout, "page_first_kv_split")

    def test_hierarchical_cache_non_mla(self):
        args = self._call(32 * 1024, enable_hierarchical_cache=True, use_mla=False)
        self.assertEqual(args.hicache_io_backend, "kernel_ascend")
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")


class TestInitNpuBackend(unittest.TestCase):
    def setUp(self):
        # _call_once sets the flag on the original fn (behind functools.wraps).
        init_npu_backend.__wrapped__._has_been_called = False

    def _npu_stubs(self):
        torch_npu = types.ModuleType("torch_npu")
        torch_npu.npu = MagicMock()
        torch_npu.npu.config = MagicMock()
        torch_npu.npu.set_compile_mode = MagicMock()
        contrib = types.ModuleType("torch_npu.contrib")
        contrib.transfer_to_npu = MagicMock()
        return torch_npu, contrib

    def test_asserts_on_non_npu(self):
        with patch.object(npu_utils, "_is_npu", False):
            with self.assertRaises(AssertionError):
                init_npu_backend()

    def test_success_on_npu(self):
        torch_npu, contrib = self._npu_stubs()
        orig_available = torch.cuda.is_available
        try:
            with patch.object(npu_utils, "_is_npu", True), patch.dict(
                sys.modules,
                {
                    "custom_ops": types.ModuleType("custom_ops"),
                    "sgl_kernel_npu": types.ModuleType("sgl_kernel_npu"),
                    "torch_npu": torch_npu,
                    "torch_npu.contrib": contrib,
                },
            ):
                init_npu_backend()
                self.assertFalse(torch.cuda.is_available())
                self.assertTrue(torch_npu.npu.config.allow_internal_format)
                torch_npu.npu.set_compile_mode.assert_called_once_with(jit_compile=False)
        finally:
            torch.cuda.is_available = orig_available

    def test_missing_custom_kernels_still_inits(self):
        # custom_ops / sgl_kernel_npu absent -> ImportError is caught and init proceeds.
        torch_npu, contrib = self._npu_stubs()
        orig_available = torch.cuda.is_available
        try:
            with patch.object(npu_utils, "_is_npu", True), patch.dict(
                sys.modules, {"torch_npu": torch_npu, "torch_npu.contrib": contrib}
            ):
                init_npu_backend()
                torch_npu.npu.set_compile_mode.assert_called_once()
        finally:
            torch.cuda.is_available = orig_available

    def test_multimodal_gen_skips_transfer(self):
        # When sglang.multimodal_gen is loaded, transfer_to_npu (and the
        # cuda.is_available / allow_internal_format mocks it performs) is
        # skipped, but set_compile_mode is still applied.
        torch_npu, contrib = self._npu_stubs()
        orig_available = torch.cuda.is_available
        try:
            with patch.object(npu_utils, "_is_npu", True), patch.dict(
                sys.modules,
                {
                    "sglang.multimodal_gen": MagicMock(),
                    "custom_ops": types.ModuleType("custom_ops"),
                    "sgl_kernel_npu": types.ModuleType("sgl_kernel_npu"),
                    "torch_npu": torch_npu,
                    "torch_npu.contrib": contrib,
                },
            ):
                init_npu_backend()
                # transfer_to_npu path was skipped:
                self.assertIs(torch.cuda.is_available, orig_available)
                self.assertIsInstance(
                    torch_npu.npu.config.allow_internal_format, MagicMock
                )
                torch_npu.npu.set_compile_mode.assert_called_once_with(jit_compile=False)
        finally:
            torch.cuda.is_available = orig_available

    def test_runs_only_once(self):
        torch_npu, contrib = self._npu_stubs()
        orig_available = torch.cuda.is_available
        try:
            with patch.object(npu_utils, "_is_npu", True), patch.dict(
                sys.modules,
                {
                    "custom_ops": types.ModuleType("custom_ops"),
                    "sgl_kernel_npu": types.ModuleType("sgl_kernel_npu"),
                    "torch_npu": torch_npu,
                    "torch_npu.contrib": contrib,
                },
            ):
                init_npu_backend()
                torch_npu.npu.set_compile_mode.reset_mock()
                init_npu_backend()
                torch_npu.npu.set_compile_mode.assert_not_called()
        finally:
            torch.cuda.is_available = orig_available


class TestIsNzAligned(unittest.TestCase):
    def test_less_than_2dims(self):
        self.assertFalse(_is_nz_aligned(torch.zeros(16)))

    def test_bf16_fp16(self):
        self.assertTrue(_is_nz_aligned(torch.zeros(4, 16, 16, dtype=torch.bfloat16)))
        self.assertTrue(_is_nz_aligned(torch.zeros(16, 16, dtype=torch.float16)))
        self.assertFalse(_is_nz_aligned(torch.zeros(16, 8, dtype=torch.float16)))

    def test_int8_fp8(self):
        self.assertTrue(_is_nz_aligned(torch.zeros(16, 32, dtype=torch.int8)))
        self.assertTrue(_is_nz_aligned(torch.zeros(16, 32, dtype=torch.float8_e4m3fn)))
        self.assertFalse(_is_nz_aligned(torch.zeros(16, 16, dtype=torch.int8)))

    def test_uint8_int32(self):
        self.assertTrue(_is_nz_aligned(torch.zeros(16, 64, dtype=torch.uint8)))
        self.assertTrue(_is_nz_aligned(torch.zeros(16, 64, dtype=torch.int32)))
        self.assertFalse(_is_nz_aligned(torch.zeros(16, 32, dtype=torch.uint8)))

    def test_unlisted_dtype_true(self):
        self.assertTrue(_is_nz_aligned(torch.zeros(3, 5, dtype=torch.float32)))
        self.assertTrue(_is_nz_aligned(torch.zeros(16, 16, dtype=torch.float64)))


class TestNpuFormatCast(unittest.TestCase):
    def _npu_on(self, disable=False):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(npu_utils, "_is_npu", True))
        stack.enter_context(
            patch.object(
                npu_utils.envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT,
                "get",
                return_value=disable,
            )
        )
        # torch.ops.npu only exists on NPU builds; provide it here.
        stack.enter_context(patch.object(torch.ops, "npu", MagicMock()))
        # warning_once is only patched onto loggers by NPU runtime init_logger.
        stack.enter_context(patch.object(npu_utils, "logger", MagicMock()))
        return stack

    def test_noop_on_non_npu(self):
        t = torch.randn(16, 16)
        with patch.object(npu_utils, "_is_npu", False):
            self.assertIs(npu_format_cast(t), t)

    def test_noop_when_disabled(self):
        t = torch.randn(16, 16)
        with self._npu_on(disable=True):
            self.assertIs(npu_format_cast(t), t)

    def test_noop_on_cpu(self):
        t = torch.randn(16, 16)
        with self._npu_on():
            self.assertIs(npu_format_cast(t), t)

    def test_noop_on_meta(self):
        t = torch.zeros(16, 16, device="meta")
        with self._npu_on():
            self.assertIs(npu_format_cast(t), t)

    def test_forward_packed_fp4_kwargs(self):
        t = _FakeTensor((16, 16))
        with self._npu_on(), patch.object(
            torch.ops.npu, "npu_format_cast", return_value="casted"
        ) as op:
            out = npu_format_cast(
                t, customize_dtype=torch.float8_e4m3fn, input_dtype=torch.uint8
            )
        self.assertEqual(out, "casted")
        op.assert_called_once_with(
            t,
            int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ),
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch.uint8,
        )

    def test_fallback_when_not_aligned(self):
        t = _FakeTensor((16, 16), dtype=torch.int8)
        with self._npu_on(), patch.object(
            torch.ops.npu, "npu_format_cast"
        ) as op:
            self.assertIs(npu_format_cast(t), t)
        op.assert_not_called()

    def test_cast_aligned_nz(self):
        t = _FakeTensor((16, 16))
        with self._npu_on(), patch.object(
            torch.ops.npu, "npu_format_cast", return_value="casted"
        ) as op:
            out = npu_format_cast(t)
        self.assertEqual(out, "casted")
        op.assert_called_once_with(t, NPUACLFormat.ACL_FORMAT_FRACTAL_NZ.value)

    def test_cast_nd_format_skips_alignment(self):
        t = _FakeTensor((3, 5))
        with self._npu_on(), patch.object(
            torch.ops.npu, "npu_format_cast", return_value="casted"
        ) as op:
            out = npu_format_cast(t, NPUACLFormat.ACL_FORMAT_ND)
        self.assertEqual(out, "casted")
        op.assert_called_once_with(t, NPUACLFormat.ACL_FORMAT_ND.value)


class TestIndexerWeightStream(_NPUModuleTestBase):
    def setUp(self):
        npu_utils.indexer_weight_stream = None
        self._save_npu()
        self.npu = MagicMock()
        self.npu.Stream.return_value = _FakeStream("indexer")
        torch.npu = self.npu

    def tearDown(self):
        npu_utils.indexer_weight_stream = None
        self._restore_npu()

    def test_lazily_creates_stream(self):
        self.assertIs(get_indexer_weight_stream(), self.npu.Stream.return_value)
        self.npu.Stream.assert_called_once()

    def test_caches_stream(self):
        first = get_indexer_weight_stream()
        second = get_indexer_weight_stream()
        self.assertIs(first, second)
        self.npu.Stream.assert_called_once()


class _ZbalBase(unittest.TestCase):
    def setUp(self):
        npu_utils.gva_is_inited = False
        self.zbal = SimpleNamespace(
            is_mix_alloc=MagicMock(return_value=False),
            switch_to_allocator=MagicMock(),
            zbal_init=MagicMock(return_value=1),
        )
        self._prev_zbal = sys.modules.get("zbal")

    def tearDown(self):
        npu_utils.gva_is_inited = False
        if self._prev_zbal is None:
            sys.modules.pop("zbal", None)
        else:
            sys.modules["zbal"] = self._prev_zbal

    def _patch_env(self, mem_size, bootstrap):
        stack = contextlib.ExitStack()
        stack.enter_context(
            patch.object(
                npu_utils.envs.SGLANG_ZBAL_LOCAL_MEM_SIZE, "get", return_value=mem_size
            )
        )
        stack.enter_context(
            patch.object(
                npu_utils.envs.SGLANG_ZBAL_BOOTSTRAP_URL,
                "get",
                return_value=bootstrap,
            )
        )
        stack.enter_context(patch.dict(sys.modules, {"zbal": self.zbal}))
        return stack


class TestInitZbal(_ZbalBase):
    def _call(self, mem_size=64, bootstrap="", do_check=True):
        with self._patch_env(mem_size, bootstrap):
            return init_zbal(4, 0, 1, do_check=do_check)

    def test_disabled_when_mem_size_zero(self):
        self.assertEqual(self._call(mem_size=0), 1)
        self.zbal.zbal_init.assert_not_called()

    def test_mix_alloc_returns_without_init(self):
        self.zbal.is_mix_alloc.return_value = True
        self.assertEqual(self._call(), 1)
        self.zbal.switch_to_allocator.assert_called_once()
        self.zbal.zbal_init.assert_not_called()

    def test_success_no_bootstrap(self):
        self.assertEqual(self._call(mem_size=64), 1)
        self.zbal.zbal_init.assert_called_once_with(4, 0, 1, 64 * (1024**2))
        self.assertTrue(npu_utils.gva_is_inited)

    def test_success_with_bootstrap(self):
        self.assertEqual(self._call(mem_size=64, bootstrap="http://zbal:8080"), 1)
        self.zbal.zbal_init.assert_called_once_with(
            4, 0, 1, 64 * (1024**2), ip_port="http://zbal:8080"
        )

    def test_exits_on_failure_when_do_check(self):
        self.zbal.zbal_init.return_value = 0
        with self.assertRaises(SystemExit):
            self._call()

    def test_no_exit_when_do_check_false(self):
        self.zbal.zbal_init.return_value = 0
        self.assertEqual(self._call(do_check=False), 0)


class TestLazyInitZbalGvaMem(_ZbalBase):
    def _call(self, free=50.0, mem_size=64 * 1024, bootstrap=""):
        with self._patch_env(mem_size, bootstrap), patch.object(
            npu_utils.envs.SGLANG_ZBAL_LOCAL_MEM_SIZE, "get", return_value=mem_size
        ), patch(
            "sglang.srt.utils.common.get_available_gpu_memory", return_value=free
        ):
            return lazy_init_zbal_gva_mem(
                device="npu", gpu_id=0, world_rank=1, world_size=2
            )

    def test_non_mix_returns_one(self):
        self.assertEqual(self._call(), 1)
        self.zbal.zbal_init.assert_not_called()

    def test_mix_alloc_inits(self):
        self.zbal.is_mix_alloc.return_value = True
        used_mb = int((61.2 - 50.0) * 1024)
        gva = 64 * 1024 - used_mb
        gva -= gva % 128
        self.assertEqual(self._call(), 1)
        self.zbal.zbal_init.assert_called_once_with(2, 0, 1, gva * (1024**2))
        self.assertTrue(npu_utils.gva_is_inited)

    def test_asserts_when_already_inited(self):
        self.zbal.is_mix_alloc.return_value = True
        npu_utils.gva_is_inited = True
        with self.assertRaises(AssertionError):
            self._call()


class TestShareRoutedStreams(_NPUModuleTestBase):
    def setUp(self):
        self._save_npu()
        set_share_stream(None)
        set_routed_stream(None)

    def tearDown(self):
        self._restore_npu()
        set_share_stream(None)
        set_routed_stream(None)

    def test_share_stream_set_get(self):
        stream = _FakeStream("share")
        set_share_stream(stream)
        self.assertIs(get_share_stream(), stream)

    def test_wait_share_stream_noop_when_unset(self):
        wait_share_stream()
        self.assertEqual(self._fake.current.waited_on, [])

    def test_wait_share_stream_makes_current_wait(self):
        stream = _FakeStream("share")
        set_share_stream(stream)
        wait_share_stream()
        self.assertIs(self._fake.current.waited_on[0], stream)

    def test_routed_stream_set_get(self):
        stream = _FakeStream("routed")
        set_routed_stream(stream)
        self.assertIs(get_routed_stream(), stream)

    def test_wait_routed_stream_noop_when_unset(self):
        wait_routed_stream()
        self.assertEqual(self._fake.current.waited_on, [])

    def test_wait_routed_stream_makes_current_wait(self):
        stream = _FakeStream("routed")
        set_routed_stream(stream)
        wait_routed_stream()
        self.assertIs(self._fake.current.waited_on[0], stream)


class TestProcessExperts(_NPUModuleTestBase):
    def setUp(self):
        self._save_npu()
        set_share_stream(None)
        set_routed_stream(None)

    def tearDown(self):
        self._restore_npu()
        set_share_stream(None)
        set_routed_stream(None)

    def test_process_shared_creates_and_uses_stream(self):
        out = process_shared_expert(torch.tensor([1.0]), lambda h: h * 2)
        self.assertEqual(out.item(), 2.0)
        stream = get_share_stream()
        self.assertIsNotNone(stream)
        self.assertIn(stream, self._fake.created)
        self.assertEqual(len(stream.waited_on), 1)
        self.assertIs(stream.waited_on[0], self._fake.current)

    def test_process_shared_reuses_stream(self):
        existing = _FakeStream("existing")
        set_share_stream(existing)
        process_shared_expert(torch.tensor([1.0]), lambda h: h)
        self.assertIs(get_share_stream(), existing)
        self.assertEqual(self._fake.created, [])

    def test_process_routed(self):
        out = process_routed_expert(
            torch.tensor([1.0]), torch.tensor([10.0]), lambda h, t: h + t
        )
        self.assertEqual(out.item(), 11.0)
        self.assertIsNotNone(get_routed_stream())


if __name__ == "__main__":
    unittest.main()
