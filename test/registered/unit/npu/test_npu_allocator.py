"""
Unit tests for sglang.srt.hardware_backend.npu.allocator_npu.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

# Import shims: sglang/__init__ pulls in the Lang API / hf_transformers_patches,
# which drag in optional deps (IPython, aiohttp, triton, ...). Keep them out so
# the allocator can be exercised on a CPU-only box with minimal mocking. These
# must be set up before any `import sglang.*` (e.g. register_npu_ci) triggers
# sglang/__init__.
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

# sgl_kernel_npu is imported lazily on the alloc_extend kernel path; register a
# stub module so the kernel-path tests can patch alloc_extend_kernel.
_npu_alloc_mod = types.ModuleType("sgl_kernel_npu.mem_cache.allocator")
_npu_alloc_mod.alloc_extend_kernel = None
sys.modules.setdefault("sgl_kernel_npu", types.ModuleType("sgl_kernel_npu"))
sys.modules.setdefault(
    "sgl_kernel_npu.mem_cache", types.ModuleType("sgl_kernel_npu.mem_cache")
)
sys.modules["sgl_kernel_npu.mem_cache.allocator"] = _npu_alloc_mod

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.allocator_npu import (  # noqa: E402
    NPUPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive  # noqa: E402

PAGE_SIZE = 8


def _make_allocator(size=64, need_sort=False):
    return NPUPagedTokenToKVPoolAllocator(
        size, PAGE_SIZE, torch.float32, "cpu", None, need_sort
    )


def _i64(*values):
    return torch.tensor(list(values), dtype=torch.int64)


class _KernelStub:
    """Mimics the grid-indexed NPU extend kernel launch signature."""

    def __init__(self, impl):
        self._impl = impl

    def __getitem__(self, grid):
        return self._run

    def _run(self, *args):
        # kernel(prefix_lens, seq_lens, last_loc, free_pages, out_indices,
        #        next_power_of_2(bs), page_size, max_num_extend_tokens)
        self._impl(args[0], args[1], args[2], args[3], args[4], args[6], "cpu")


class TestInit(unittest.TestCase):
    def test_construction(self):
        alloc = _make_allocator(size=64)
        self.assertEqual(alloc.num_pages, 8)
        self.assertEqual(alloc.roundup, PAGE_SIZE - 1)
        self.assertTrue(alloc.is_not_in_free_group)
        self.assertTrue(torch.equal(alloc.free_pages, torch.arange(1, 9)))
        self.assertEqual(alloc.release_pages.numel(), 0)


class TestAllocExtend(unittest.TestCase):
    def test_naive_path_fresh_sequence(self):
        # 1600 tokens -> 200 new pages, which takes the naive (>= 200) path.
        alloc = _make_allocator(size=16384)
        zero = torch.zeros(1, dtype=torch.int64)
        out = alloc.alloc_extend(zero, zero, _i64(1600), _i64(1600), _i64(-1), 1600)
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(torch.equal(out, torch.arange(8, 1608, dtype=torch.int32)))
        self.assertEqual(alloc.free_pages[0].item(), 201)

    def test_kernel_path_matches_naive(self):
        # 160 tokens -> 20 new pages, which takes the kernel (< 200) path.
        alloc = _make_allocator(size=16384)
        zero = torch.zeros(1, dtype=torch.int64)
        with patch.object(
            _npu_alloc_mod, "alloc_extend_kernel", _KernelStub(alloc_extend_naive)
        ):
            out = alloc.alloc_extend(zero, zero, _i64(160), _i64(160), _i64(-1), 160)
        self.assertTrue(torch.equal(out, torch.arange(8, 168, dtype=torch.int32)))
        self.assertEqual(alloc.free_pages[0].item(), 21)

    def test_multi_seq_kernel_path(self):
        alloc = _make_allocator(size=16384)
        prefix = _i64(0, 3)
        seq_lens = _i64(1600, 1603)
        last_loc = _i64(-1, 2)
        with patch.object(
            _npu_alloc_mod, "alloc_extend_kernel", _KernelStub(alloc_extend_naive)
        ):
            out = alloc.alloc_extend(prefix, prefix, seq_lens, seq_lens, last_loc, 3200)
        self.assertEqual(out.numel(), 3200)
        self.assertTrue(torch.equal(out[:5], torch.arange(8, 13, dtype=torch.int32)))
        # seq 1 (prefix 3) continues its partial page with tokens [3..7]...
        self.assertTrue(
            torch.equal(out[1600:1605], torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32))
        )
        # ...then finishes its last partial page with tokens [3200..3202].
        self.assertTrue(
            torch.equal(out[-3:], torch.tensor([3200, 3201, 3202], dtype=torch.int32))
        )
        self.assertEqual(len(torch.unique(out)), 3200)  # no overlaps
        self.assertEqual(alloc.free_pages[0].item(), 401)

    def test_num_new_pages_override(self):
        alloc = _make_allocator(size=16384)
        zero = torch.zeros(1, dtype=torch.int64)
        # Explicit override: only 2 pages are consumed regardless of seq_lens.
        with patch.object(
            _npu_alloc_mod, "alloc_extend_kernel", _KernelStub(alloc_extend_naive)
        ):
            alloc.alloc_extend(zero, zero, _i64(1600), _i64(1600), _i64(-1), 1600, num_new_pages=2)
        self.assertEqual(alloc.free_pages[0].item(), 3)

    def test_returns_none_when_insufficient(self):
        alloc = _make_allocator(size=64)
        zero = torch.zeros(1, dtype=torch.int64)
        self.assertIsNone(
            alloc.alloc_extend(zero, zero, _i64(1600), _i64(1600), _i64(-1), 1600)
        )

    def test_need_sort_merges_release_pages(self):
        # 200 released pages + empty free list -> merge -> naive extend succeeds.
        alloc = _make_allocator(size=16384, need_sort=True)
        alloc.free_pages = torch.empty((0,), dtype=torch.int64)
        alloc.release_pages = torch.arange(1, 201, dtype=torch.int64)
        zero = torch.zeros(1, dtype=torch.int64)
        out = alloc.alloc_extend(zero, zero, _i64(1600), _i64(1600), _i64(-1), 1600)
        self.assertTrue(torch.equal(out, torch.arange(8, 1608, dtype=torch.int32)))
        self.assertEqual(alloc.release_pages.numel(), 0)
        self.assertEqual(alloc.free_pages.numel(), 0)  # all 200 pages were used

    def test_debug_mode_asserts_last_loc_consistency(self):
        with patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"}):
            alloc = _make_allocator(size=16384)
            zero = torch.zeros(1, dtype=torch.int64)
            with self.assertRaises(AssertionError):
                alloc.alloc_extend(zero, zero, _i64(8), _i64(8), _i64(0), 8)


class TestAllocDecode(unittest.TestCase):
    def test_continue_existing_page(self):
        alloc = _make_allocator(size=64)
        # seq_len 4 keeps us on the current page, so no new page is allocated.
        out = alloc.alloc_decode(_i64(4), _i64(4), _i64(10))
        self.assertEqual(out.item(), 11)
        self.assertEqual(alloc.free_pages[0].item(), 1)  # unchanged

    def test_new_page(self):
        alloc = _make_allocator(size=64)
        # seq_len 9 starts a new page: token = first slot of the next free page.
        out = alloc.alloc_decode(_i64(9), _i64(9), _i64(-1))
        self.assertEqual(out.item(), 8)
        self.assertEqual(alloc.free_pages[0].item(), 2)

    def test_mixed_batch(self):
        alloc = _make_allocator(size=64)
        out = alloc.alloc_decode(_i64(9, 4), _i64(9, 4), _i64(-1, 10))
        self.assertTrue(torch.equal(out, torch.tensor([8, 11], dtype=torch.int32)))
        self.assertEqual(alloc.free_pages[0].item(), 2)

    def test_returns_none_when_insufficient(self):
        alloc = _make_allocator(size=64)
        # 9 sequences each needing a new page, but only 8 pages are free.
        seq_lens = torch.full((9,), 9, dtype=torch.int64)
        self.assertIsNone(alloc.alloc_decode(seq_lens, seq_lens, torch.full((9,), -1, dtype=torch.int64)))

    def test_need_sort_merges_release_pages(self):
        alloc = _make_allocator(size=64, need_sort=True)
        alloc.free_pages = torch.empty((0,), dtype=torch.int64)
        alloc.release_pages = _i64(1)
        out = alloc.alloc_decode(_i64(9), _i64(9), _i64(-1))
        self.assertEqual(out.item(), 8)
        self.assertEqual(alloc.free_pages.numel(), 0)  # the single page was used
        self.assertEqual(alloc.release_pages.numel(), 0)

    def test_debug_mode_asserts_seq_len_consistency(self):
        with patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"}):
            alloc = _make_allocator(size=64)
            with self.assertRaises(AssertionError):
                alloc.alloc_decode(_i64(5), _i64(5), _i64(10))


class TestFree(unittest.TestCase):
    def test_empty_is_noop(self):
        alloc = _make_allocator(size=64)
        alloc.free(torch.empty(0, dtype=torch.int64))
        self.assertTrue(torch.equal(alloc.free_pages, torch.arange(1, 9)))

    def test_reclaims_pages_with_dedup(self):
        alloc = _make_allocator(size=64)
        alloc.alloc(16)  # pages 1 and 2 are now in use
        self.assertEqual(alloc.free_pages[0].item(), 3)
        # Tokens from pages 1 and 2 -> both pages pushed back to the free list.
        alloc.free(_i64(8, 16, 20))
        self.assertTrue(torch.equal(alloc.free_pages[:3], _i64(1, 2, 3)))

    def test_need_sort_routes_to_release_pages(self):
        alloc = _make_allocator(size=64, need_sort=True)
        alloc.free(_i64(8))
        self.assertTrue(torch.equal(alloc.free_pages, torch.arange(1, 9)))  # untouched
        self.assertTrue(torch.equal(alloc.release_pages, _i64(1)))

    def test_free_group_is_deferred(self):
        alloc = _make_allocator(size=64)
        alloc.alloc(8)  # page 1 is now in use
        self.assertEqual(alloc.free_pages[0].item(), 2)
        alloc.free_group_begin()
        alloc.free(_i64(8))
        # Not reclaimed while inside the group...
        self.assertEqual(alloc.free_pages[0].item(), 2)
        self.assertEqual(len(alloc.free_group), 1)
        # ...and reclaimed on free_group_end.
        alloc.free_group_end()
        self.assertEqual(alloc.free_pages[0].item(), 1)

    def test_debug_mode_asserts_unique_free_pages(self):
        with patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"}):
            alloc = _make_allocator(size=64)
            alloc.free_pages = _i64(1, 1)  # already duplicated
            with self.assertRaises(AssertionError):
                alloc.free(_i64(8))


if __name__ == "__main__":
    unittest.main()
