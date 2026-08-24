"""
Unit tests for sglang.srt.hardware_backend.npu.memory_pool_npu.
"""

import os
import sys
import types
import unittest
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

# sglang.srt.utils.network imports zmq (only used by distributed comms at
# runtime). Stub it so the import chain from memory_pool_npu resolves on a
# box without pyzmq installed.
sys.modules.setdefault("zmq", MagicMock())

# sglang.kernels.ops.quantization.fp8_kernel (imported by the MHA pool base
# class) pulls in sglang.srt.layers.deep_gemm_wrapper, whose CUDA-only init
# chain (torch.cuda internals / pynccl_allocator) cannot load on this box.
# The wrapper is only referenced from Hopper/Blackwell paths we never hit.
sys.modules.setdefault("sglang.srt.layers.deep_gemm_wrapper", MagicMock())

# sglang.srt.configs eagerly registers every config in its __init__, which
# trips over the installed transformers version (e.g. "qwen3_asr" already
# registered). We register a stub package (with the real __path__ so light
# std-lib-only submodules like embedding_model_spec still load) and stub the
# heavier mamba_utils, which the MHA pool only needs for a lazy annotation.
_SGLANG_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python", "sglang")
)
_configs_stub = types.ModuleType("sglang.srt.configs")
_configs_stub.__path__ = [os.path.join(_SGLANG_SRC, "srt", "configs")]
sys.modules.setdefault("sglang.srt.configs", _configs_stub)
_configs_mamba_utils = types.ModuleType("sglang.srt.configs.mamba_utils")
_configs_mamba_utils.BaseLinearStateParams = MagicMock()
sys.modules.setdefault("sglang.srt.configs.mamba_utils", _configs_mamba_utils)

# sglang.srt.layers.quantization.__init__ eagerly imports every quant recipe
# (awq -> linear -> moe -> ...), which needs CUDA-only code. Bypass the parent
# init and load only fp4_kv_cache_quant_method (a std-lib-only module) so the
# MHA pool base class gets the real UnquantizedKVCacheMethod.
_quant_stub = types.ModuleType("sglang.srt.layers.quantization")
_quant_stub.__path__ = [os.path.join(_SGLANG_SRC, "srt", "layers", "quantization")]
sys.modules.setdefault("sglang.srt.layers.quantization", _quant_stub)

# kvfp4_tensor defines @torch.compile-decorated helpers at import time, which
# drags torch._dynamo -> torch._inductor -> triton.compiler (not installed on
# this box). fp4_kv_cache_quant_method only needs E2M1_MAX from it and the
# compile paths are never exercised by these tests, so stub the module.
_kvfp4_stub = types.ModuleType("sglang.srt.layers.quantization.kvfp4_tensor")
_kvfp4_stub.E2M1_MAX = 6.0
sys.modules["sglang.srt.layers.quantization.kvfp4_tensor"] = _kvfp4_stub

# sglang.srt.layers.attention.dsa.utils drags in dp_attention ->
# pynccl_allocator, which needs torch.cuda.memory internals unavailable here.
# aiter_can_use_preshuffle_paged_mqa gates the aiter import in
# index_buf_accessor (aiter is Ascend/CUDA-only, not installed here); it must
# return a falsy value so that branch is skipped.
_dsa_utils = types.ModuleType("sglang.srt.layers.attention.dsa.utils")
_dsa_utils.INDEXER_K_CACHE_PRESHUFFLE_TILE = 64
_dsa_utils.aiter_can_use_preshuffle_paged_mqa = lambda: False
sys.modules.setdefault("sglang.srt.layers.attention.dsa.utils", _dsa_utils)

# sglang.srt.distributed.device_communicators.pynccl_allocator imports CUDA
# allocator hooks (torch.cuda.memory._cuda_*Pool) missing from this torch build.
# It is only used by NCCL / symmetric-memory paths we never touch at runtime.
sys.modules.setdefault(
    "sglang.srt.distributed.device_communicators.pynccl_allocator",
    MagicMock(),
)

# sglang.kernels.jit.utils.compile.loader imports fcntl (Linux-only). Provide a
# stub so the module chain imports on Windows, where the real module is absent.
try:
    import fcntl  # noqa: F401
except ImportError:
    _fcntl = types.ModuleType("fcntl")
    _fcntl.flock = lambda *a, **kw: None
    _fcntl.LOCK_EX = 2
    _fcntl.LOCK_SH = 1
    _fcntl.LOCK_UN = 8
    _fcntl.LOCK_NB = 4
    sys.modules["fcntl"] = _fcntl

try:
    import triton  # noqa: F401
except ImportError:
    _triton = types.ModuleType("triton")
    _triton.jit = MagicMock(return_value=lambda f: f)
    _triton.autotune = lambda *a, **kw: (lambda f: f)
    _triton.Config = MagicMock()
    sys.modules["triton"] = _triton
    sys.modules.setdefault("triton.language", MagicMock())
    sys.modules.setdefault("triton.backends", MagicMock())

# sgl_kernel_npu is imported lazily inside set_kv_buffer_prefix_valid; register
# a stub module so the triton-store path can be patched and verified.
sys.modules.setdefault("sgl_kernel_npu", types.ModuleType("sgl_kernel_npu"))
sys.modules.setdefault(
    "sgl_kernel_npu.mem_cache", types.ModuleType("sgl_kernel_npu.mem_cache")
)
_npu_kv_store_mod = types.ModuleType("sgl_kernel_npu.mem_cache.kv_cache_store")
_npu_kv_store_mod.store_kv_cache_prefix_valid_npu_triton = None
sys.modules.setdefault("sgl_kernel_npu.mem_cache.kv_cache_store", _npu_kv_store_mod)

# torch_npu is only imported by memory_pool_npu when is_npu() is True (never on
# this CPU box), and every call site resolves torch_npu.* through the module
# namespace. We inject a stub into that namespace below so the KV-write paths
# can be patched and verified. Note: it must NOT be registered in sys.modules,
# otherwise importlib.util.find_spec("torch_npu") (used by transformers' lazy
# import machinery) raises "torch_npu.__spec__ is None".
_torch_npu_mod = types.ModuleType("torch_npu")
_torch_npu_mod.npu_scatter_nd_update_ = MagicMock()
_torch_npu_mod._npu_reshape_and_cache = MagicMock()

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.memory_pool_npu import (  # noqa: E402
    NPUMHATokenToKOnlyPool,
    NPUMHATokenToKVPool,
    NPUMiniMaxSparseKVPool,
    NPUMLATokenToKVPool,
    _init_npu_conv_state,
)

_mem_pool_mod = sys.modules["sglang.srt.hardware_backend.npu.memory_pool_npu"]
_mem_pool_mod.torch_npu = _torch_npu_mod

SIZE, PAGE, H, D = 64, 8, 2, 8
NUM_PAGES = SIZE // PAGE + 1  # 9: padded slot 0


class _Layer:
    """Minimal RadixAttention stand-in exposing only layer_id."""

    def __init__(self, layer_id):
        self.layer_id = layer_id


class _FakeNPU:
    """torch.npu stand-in for the CPU-copy paths."""

    def synchronize(self):
        pass


def _scatter_nd_update_cpu(data, indices, updates):
    data[indices.reshape(-1)] = updates


def _reshape_and_cache_cpu(*, key, value, key_cache, value_cache, slot_indices):
    key_cache.view(-1, key.shape[-2], key.shape[-1])[slot_indices] = key
    value_cache.view(-1, value.shape[-2], value.shape[-1])[slot_indices] = value


def _make_mha(use_fia=False, layer_num=2, enable_kv_cache_copy=False, prefix_triton=False):
    env = {}
    if use_fia:
        env["ASCEND_USE_FIA"] = "1"
    if prefix_triton:
        env["SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE"] = "1"
    with patch.dict(os.environ, env, clear=False):
        return NPUMHATokenToKVPool(
            size=SIZE,
            page_size=PAGE,
            dtype=torch.float32,
            head_num=H,
            head_dim=D,
            layer_num=layer_num,
            device="cpu",
            enable_memory_saver=False,
            v_head_dim=D,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )


def _make_konly(use_fia=False, layer_num=2):
    with patch.dict(
        os.environ, {"ASCEND_USE_FIA": "1"} if use_fia else {}, clear=False
    ):
        return NPUMHATokenToKOnlyPool(
            size=SIZE,
            page_size=PAGE,
            dtype=torch.float32,
            head_num=H,
            head_dim=D,
            layer_num=layer_num,
            device="cpu",
            enable_memory_saver=False,
        )


class TestConvStateInit(unittest.TestCase):
    def test_mamba_shape(self):
        inp = torch.zeros(2, 3)
        out = _init_npu_conv_state(inp, [(4, 5)])
        self.assertEqual(len(out), 1)
        self.assertEqual(tuple(out[0].shape), (2, 3, 5, 4))

    def test_kda_shape(self):
        inp = torch.zeros(2, 3)
        out = _init_npu_conv_state(inp, [(4, 5)], is_kda=True)
        self.assertEqual(tuple(out[0].shape), (2, 3, 5, 4))

    def test_speculative_draft_tokens(self):
        inp = torch.zeros(2, 3)
        out = _init_npu_conv_state(inp, [(4, 5)], speculative_num_draft_tokens=3)
        self.assertEqual(tuple(out[0].shape), (2, 3, 7, 4))

    def test_multiple_shapes(self):
        inp = torch.zeros(2, 3)
        out = _init_npu_conv_state(inp, [(4, 5), (6, 7)])
        self.assertEqual(
            [tuple(t.shape) for t in out], [(2, 3, 5, 4), (2, 3, 7, 6)]
        )


class TestMHAPoolInit(unittest.TestCase):
    def test_construction_non_fia(self):
        pool = _make_mha()
        self.assertFalse(pool.use_fia)
        self.assertIsInstance(pool.k_buffer, torch.Tensor)
        self.assertEqual(tuple(pool.k_buffer.shape), (2, NUM_PAGES, PAGE, H, D))
        self.assertEqual(tuple(pool.v_buffer.shape), (2, NUM_PAGES, PAGE, H, D))
        self.assertEqual(pool.kv_cache_layout, "nhd")
        self.assertIsNone(pool.alt_stream)

    def test_construction_fia(self):
        pool = _make_mha(use_fia=True)
        self.assertTrue(pool.use_fia)
        self.assertIsInstance(pool.k_buffer, list)
        self.assertEqual(len(pool.k_buffer), 2)
        self.assertEqual(tuple(pool.k_buffer[0].shape), (NUM_PAGES * PAGE, 1, H, D))
        self.assertEqual(tuple(pool.v_buffer[0].shape), (NUM_PAGES * PAGE, 1, H, D))

    def test_kv_copy_config_disabled(self):
        pool = _make_mha(enable_kv_cache_copy=True)
        self.assertIsNone(pool._kv_copy_config)

    def test_get_key_value_buffers(self):
        pool = _make_mha()
        for i in range(2):
            self.assertTrue(torch.equal(pool.get_key_buffer(i), pool.k_buffer[i]))
            self.assertTrue(torch.equal(pool.get_value_buffer(i), pool.v_buffer[i]))


class TestMHASetKvBuffer(unittest.TestCase):
    def test_non_fia_uses_reshape_and_cache(self):
        pool = _make_mha()
        loc = torch.tensor([3, 10], dtype=torch.int64)
        k = torch.randn(2, H, D)
        v = torch.randn(2, H, D)
        with patch.object(_torch_npu_mod, "_npu_reshape_and_cache") as m:
            pool.set_kv_buffer(_Layer(1), loc, k, v)
        m.assert_called_once()
        kw = m.call_args.kwargs
        self.assertTrue(torch.equal(kw["key"], k))
        self.assertTrue(torch.equal(kw["value"], v))
        self.assertTrue(torch.equal(kw["slot_indices"], loc.to(torch.int32)))
        self.assertTrue(
            torch.equal(kw["key_cache"], pool.k_buffer[0].view(-1, PAGE, H, D))
        )
        self.assertTrue(
            torch.equal(kw["value_cache"], pool.v_buffer[0].view(-1, PAGE, H, D))
        )

    def test_non_fia_writes_values(self):
        pool = _make_mha()
        loc = torch.tensor([5, 12])
        k = torch.randn(2, H, D)
        v = torch.randn(2, H, D)
        with patch.object(_torch_npu_mod, "_npu_reshape_and_cache", _reshape_and_cache_cpu):
            pool.set_kv_buffer(_Layer(0), loc, k, v)
        self.assertTrue(torch.equal(pool.k_buffer[0].view(-1, H, D)[loc], k))
        self.assertTrue(torch.equal(pool.v_buffer[0].view(-1, H, D)[loc], v))

    def test_fia_scatter(self):
        pool = _make_mha(use_fia=True)
        loc = torch.tensor([4, 20])
        k = torch.randn(2, H, D)
        v = torch.randn(2, H, D)
        with patch.object(_torch_npu_mod, "npu_scatter_nd_update_", _scatter_nd_update_cpu):
            pool.set_kv_buffer(_Layer(0), loc, k, v)
        k_buf = pool.k_buffer[0].view(-1, H, D)
        v_buf = pool.v_buffer[0].view(-1, H, D)
        self.assertTrue(torch.equal(k_buf[loc], k))
        self.assertTrue(torch.equal(v_buf[loc], v))

    def test_fia_numel_mismatch_raises(self):
        pool = _make_mha(use_fia=True)
        with self.assertRaises(ValueError):
            pool.set_kv_buffer(_Layer(0), torch.tensor([1, 2]), torch.randn(1, H, D), torch.randn(2, H, D))

    def test_scale_and_dtype_cast(self):
        pool = _make_mha()
        loc = torch.tensor([1, 2])
        k = torch.randn(2, H, D, dtype=torch.float64)
        v = torch.randn(2, H, D, dtype=torch.float64)
        # div_() scales cache_k/cache_v in place, so snapshot the expectation first.
        expected_k = (k / 2.0).to(torch.float32)
        expected_v = (v / 4.0).to(torch.float32)
        with patch.object(_torch_npu_mod, "_npu_reshape_and_cache", _reshape_and_cache_cpu):
            pool.set_kv_buffer(_Layer(0), loc, k, v, k_scale=2.0, v_scale=4.0)
        self.assertTrue(torch.allclose(pool.k_buffer[0].view(-1, H, D)[loc], expected_k))
        self.assertTrue(torch.allclose(pool.v_buffer[0].view(-1, H, D)[loc], expected_v))

    def test_layer_id_override(self):
        pool = _make_mha()
        loc = torch.tensor([2])
        k = torch.randn(1, H, D)
        v = torch.randn(1, H, D)
        with patch.object(_torch_npu_mod, "_npu_reshape_and_cache", _reshape_and_cache_cpu):
            pool.set_kv_buffer(_Layer(0), loc, k, v, layer_id_override=1)
        self.assertTrue(torch.equal(pool.k_buffer[1].view(-1, H, D)[[2]], k))
        self.assertFalse(torch.equal(pool.k_buffer[0].view(-1, H, D)[[2]], k))


class TestMHAContiguousBufInfos(unittest.TestCase):
    def test_non_fia(self):
        pool = _make_mha()
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 4)  # k+v per layer
        self.assertEqual(ptrs[:2], [pool.k_buffer[0].data_ptr(), pool.k_buffer[1].data_ptr()])
        self.assertEqual(lens[:2], [pool.k_buffer[0].nbytes, pool.k_buffer[1].nbytes])
        self.assertEqual(item_lens[:2], [pool.k_buffer[0][0].nbytes, pool.k_buffer[1][0].nbytes])

    def test_fia(self):
        pool = _make_mha(use_fia=True)
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 4)
        self.assertEqual(item_lens[0], pool.k_buffer[0][0].nbytes * PAGE)
        self.assertEqual(lens[0], pool.k_buffer[0].nbytes)


class TestMHACpuCopy(unittest.TestCase):
    def test_get_cpu_copy(self):
        pool = _make_mha()
        pool.k_buffer[1].view(-1, H, D)[2, 0, 0] = 1.0
        pool.k_buffer[1].view(-1, H, D)[5, 1, 1] = 2.0
        with patch.object(torch, "npu", _FakeNPU(), create=True):
            cpu = pool.get_cpu_copy(torch.tensor([2, 5]))
        k_cpu, v_cpu = cpu[1][0][0], cpu[1][0][1]
        self.assertEqual(tuple(k_cpu.shape), (2, H, D))
        self.assertEqual(k_cpu[0, 0, 0].item(), 1.0)
        self.assertEqual(k_cpu[1, 1, 1].item(), 2.0)
        self.assertTrue(torch.equal(v_cpu, torch.zeros(2, H, D)))

    def test_load_cpu_copy(self):
        pool = _make_mha()
        indices = torch.tensor([1, 3])
        kv_cache_cpu = [
            [[torch.full((2, H, D), 1.0 + layer), torch.full((2, H, D), 10.0 + layer)]]
            for layer in range(2)
        ]
        with patch.object(torch, "npu", _FakeNPU(), create=True):
            pool.load_cpu_copy(kv_cache_cpu, indices)
        self.assertTrue(torch.equal(pool.k_buffer[0].view(-1, H, D)[indices], torch.full((2, H, D), 1.0)))
        self.assertTrue(torch.equal(pool.v_buffer[0].view(-1, H, D)[indices], torch.full((2, H, D), 10.0)))
        self.assertTrue(torch.equal(pool.k_buffer[1].view(-1, H, D)[indices], torch.full((2, H, D), 2.0)))


class TestMHAPrefixValid(unittest.TestCase):
    def test_fallback_without_triton_store(self):
        pool = _make_mha()
        loc_2d = torch.tensor([[2, 3], [5, 6]])
        commit_lens = torch.tensor([2, 1], dtype=torch.int32)
        k = torch.randn(4, H, D)
        v = torch.randn(4, H, D)
        with patch.object(_torch_npu_mod, "_npu_reshape_and_cache", _reshape_and_cache_cpu):
            pool.set_kv_buffer_prefix_valid(_Layer(0), loc_2d, commit_lens, k, v)
        k_view = pool.k_buffer[0].view(-1, H, D)
        self.assertTrue(torch.equal(k_view[2], k[0]))
        self.assertTrue(torch.equal(k_view[3], k[1]))
        self.assertTrue(torch.equal(k_view[5], k[2]))
        self.assertFalse(torch.equal(k_view[6], k[3]))

    def test_triton_store_path(self):
        pool = _make_mha(prefix_triton=True)
        loc_2d = torch.tensor([[2, 3], [5, 6]])
        commit_lens = torch.tensor([2, 2], dtype=torch.int32)
        k = torch.randn(4, H, D)
        v = torch.randn(4, H, D)
        with patch(
            "sgl_kernel_npu.mem_cache.kv_cache_store.store_kv_cache_prefix_valid_npu_triton"
        ) as m:
            pool.set_kv_buffer_prefix_valid(_Layer(0), loc_2d, commit_lens, k, v)
        m.assert_called_once()
        self.assertEqual(pool._debug_prefix_valid_backend, "npu_triton")

    def test_triton_store_rejects_1d_loc(self):
        pool = _make_mha(prefix_triton=True)
        with self.assertRaises(ValueError):
            pool.set_kv_buffer_prefix_valid(
                _Layer(0),
                torch.tensor([1, 2]),
                torch.tensor([1]),
                torch.randn(2, H, D),
                torch.randn(2, H, D),
            )


class TestKOnlyPool(unittest.TestCase):
    def test_construction(self):
        pool = _make_konly()
        self.assertEqual(tuple(pool.k_buffer.shape), (2, NUM_PAGES, PAGE, H, D))
        self.assertEqual(pool.get_kv_size_bytes(), (2 * NUM_PAGES * PAGE * H * D * 4, 0))

    def test_construction_fia(self):
        pool = _make_konly(use_fia=True)
        self.assertTrue(pool.use_fia)
        self.assertIsInstance(pool.k_buffer, list)
        self.assertEqual(tuple(pool.k_buffer[0].shape), (NUM_PAGES * PAGE, 1, H, D))

    def test_set_k_buffer(self):
        pool = _make_konly()
        loc = torch.tensor([2, 7])
        k = torch.randn(2, H, D)
        with patch.object(_torch_npu_mod, "npu_scatter_nd_update_", _scatter_nd_update_cpu):
            pool.set_k_buffer(0, loc, k)
        self.assertTrue(torch.equal(pool.k_buffer[0].view(-1, H, D)[loc], k))

    def test_get_contiguous_buf_infos(self):
        pool = _make_konly()
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(item_lens[0], pool.k_buffer[0][0].nbytes)
        self.assertEqual(lens[0], pool.k_buffer[0].nbytes)

    def test_no_value_buffer(self):
        pool = _make_konly()
        with self.assertRaises(NotImplementedError):
            pool.get_value_buffer(0)


class TestMiniMaxSparseKVPool(unittest.TestCase):
    def _make(self, disable_sparse=True):
        disable = [1, 2] if disable_sparse else []
        return NPUMiniMaxSparseKVPool(
            size=SIZE,
            page_size=PAGE,
            dtype=torch.float32,
            head_num=H,
            head_dim=D,
            idx_head_dim=4,
            dense_layer_ids=[0],
            sparse_layer_ids=[1, 2],
            device="cpu",
            disable_value_sparse_layer_ids=disable,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=3,
        )

    def test_construction_npu_pools(self):
        pool = self._make()
        self.assertIsInstance(pool.main_pool, NPUMHATokenToKVPool)
        self.assertIsNone(pool.index_kv_pool)
        self.assertIsInstance(pool.index_k_pool, NPUMHATokenToKOnlyPool)

    def test_construction_kv_index_pool(self):
        pool = self._make(disable_sparse=False)
        self.assertIsInstance(pool.index_kv_pool, NPUMHATokenToKVPool)
        self.assertIsNone(pool.index_k_pool)

    def test_get_index_k_state_buf_infos(self):
        pool = self._make()
        ptrs, lens, item_lens = pool.get_index_k_state_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(item_lens[0], pool.index_k_pool.get_key_buffer(0)[0].nbytes)
        self.assertEqual(lens[0], pool.index_k_pool.get_key_buffer(0).nbytes)


class TestMLAPool(unittest.TestCase):
    KV, ROPE, IDX = 4, 2, 4

    def _make(self, index_head_dim=None, layer_num=2):
        return NPUMLATokenToKVPool(
            size=SIZE,
            page_size=PAGE,
            dtype=torch.float32,
            kv_lora_rank=self.KV,
            qk_rope_head_dim=self.ROPE,
            layer_num=layer_num,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=index_head_dim,
        )

    def test_construction(self):
        pool = self._make()
        self.assertEqual(tuple(pool.k_buffer.shape), (2, NUM_PAGES, PAGE, 1, self.KV))
        self.assertEqual(tuple(pool.v_buffer.shape), (2, NUM_PAGES, PAGE, 1, self.ROPE))
        self.assertIsNone(pool.index_k_buffer)

    def test_construction_with_index_head_dim(self):
        pool = self._make(index_head_dim=self.IDX)
        self.assertEqual(tuple(pool.index_k_buffer.shape), (2, NUM_PAGES, PAGE, 1, self.IDX))

    def test_set_kv_buffer(self):
        pool = self._make()
        loc = torch.tensor([3, 9])
        k = torch.randn(2, 1, self.KV)
        v = torch.randn(2, 1, self.ROPE)
        with patch.object(_torch_npu_mod, "npu_scatter_nd_update_", _scatter_nd_update_cpu):
            pool.set_kv_buffer(_Layer(0), loc, k, v)
        self.assertTrue(torch.equal(pool.k_buffer[0].view(-1, 1, self.KV)[loc], k))
        self.assertTrue(torch.equal(pool.v_buffer[0].view(-1, 1, self.ROPE)[loc], v))

    def test_set_kv_buffer_splits_combined_cache(self):
        pool = self._make()
        loc = torch.tensor([1, 2])
        k = torch.randn(2, 1, self.KV)
        v = torch.randn(2, 1, self.ROPE)
        combined = torch.cat([k, v], dim=-1)
        with patch.object(_torch_npu_mod, "npu_scatter_nd_update_", _scatter_nd_update_cpu):
            pool.set_kv_buffer(_Layer(0), loc, combined, None)
        self.assertTrue(torch.equal(pool.k_buffer[0].view(-1, 1, self.KV)[loc], k))
        self.assertTrue(torch.equal(pool.v_buffer[0].view(-1, 1, self.ROPE)[loc], v))

    def test_set_index_k_buffer(self):
        pool = self._make(index_head_dim=self.IDX)
        loc = torch.tensor([4, 7])
        ik = torch.randn(2, 1, self.IDX)
        with patch.object(_torch_npu_mod, "npu_scatter_nd_update_", _scatter_nd_update_cpu):
            pool.set_index_k_buffer(0, loc, ik)
        self.assertTrue(torch.equal(pool.index_k_buffer[0].view(-1, 1, self.IDX)[loc], ik))

    def test_get_state_buf_infos(self):
        pool_no = self._make()
        self.assertEqual(pool_no.get_state_buf_infos(), ([], [], []))
        pool = self._make(index_head_dim=self.IDX)
        ptrs, lens, item_lens = pool.get_state_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(item_lens[0], pool.index_k_buffer[0][0].nbytes)

    def test_get_contiguous_buf_infos(self):
        pool = self._make(index_head_dim=self.IDX)
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 6)  # k+v+ik per layer
        self.assertEqual(item_lens[0], pool.k_buffer[0][0].nbytes)
        self.assertEqual(item_lens[2], pool.v_buffer[0][0].nbytes)
        self.assertEqual(item_lens[4], pool.index_k_buffer[0][0].nbytes)

    def test_get_kv_size_bytes(self):
        pool = self._make(index_head_dim=self.IDX)
        expected = sum(
            t.numel() * t.element_size()
            for t in (*pool.k_buffer, *pool.v_buffer, *pool.index_k_buffer)
        )
        self.assertEqual(pool.get_kv_size_bytes(), expected)

    def test_get_cpu_copy_load_roundtrip(self):
        pool = self._make(index_head_dim=self.IDX)
        pool.k_buffer[0].view(-1, 1, self.KV)[2, 0, 0] = 7.0
        with patch.object(torch, "npu", _FakeNPU(), create=True):
            cpu = pool.get_cpu_copy(torch.tensor([2]))
        self.assertEqual(len(cpu[0][0]), 3)  # k, v, ik
        self.assertEqual(cpu[0][0][0][0, 0, 0].item(), 7.0)

        fresh = self._make(index_head_dim=self.IDX)
        with patch.object(torch, "npu", _FakeNPU(), create=True):
            fresh.load_cpu_copy(cpu, torch.tensor([2]))
        self.assertEqual(fresh.k_buffer[0].view(-1, 1, self.KV)[2, 0, 0].item(), 7.0)


if __name__ == "__main__":
    unittest.main()
