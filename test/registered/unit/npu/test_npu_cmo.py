"""
Unit tests for sglang.srt.hardware_backend.npu.cmo.
"""

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

try:
    import triton  # noqa: F401
except ImportError:
    _triton = types.ModuleType("triton")
    _triton.jit = MagicMock(return_value=lambda f: f)
    _triton.autotune = lambda *a, **kw: (lambda f: f)
    sys.modules["triton"] = _triton
    sys.modules.setdefault("triton.language", MagicMock())
    sys.modules.setdefault("triton.backends", MagicMock())

# torch_npu is imported lazily inside prepare_weight_cache; register a stub so
# the prefetch call can be patched and verified.
_torch_npu_mod = types.ModuleType("torch_npu")
_torch_npu_mod.npu_prefetch = MagicMock()
sys.modules.setdefault("torch_npu", _torch_npu_mod)

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.cmo import (  # noqa: E402
    get_cmo_stream,
    get_share_stream,
    prepare_weight_cache,
    set_cmo_stream,
    set_share_stream,
    shared_expert_on_independent_stream,
    wait_cmo_stream,
    wait_share_stream,
)

DEFAULT_PREFETCH_MAX_SIZE = 1000000000


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


class _NPUStreamTestBase(unittest.TestCase):
    def setUp(self):
        self.npu = _FakeNPU()
        self._prev_npu = getattr(torch, "npu", None)
        torch.npu = self.npu
        set_cmo_stream(None)
        set_share_stream(None)

    def tearDown(self):
        if self._prev_npu is None:
            del torch.npu
        else:
            torch.npu = self._prev_npu
        set_cmo_stream(None)
        set_share_stream(None)


class TestCmoStream(_NPUStreamTestBase):
    def test_get_cmo_stream_returns_none_after_reset(self):
        self.assertIsNone(get_cmo_stream())

    def test_set_and_get_cmo_stream(self):
        stream = _FakeStream("cmo")
        set_cmo_stream(stream)
        self.assertIs(get_cmo_stream(), stream)

    def test_wait_cmo_stream_noop_when_unset(self):
        wait_cmo_stream()
        self.assertEqual(self.npu.current.waited_on, [])

    def test_wait_cmo_stream_makes_current_wait(self):
        stream = _FakeStream("cmo")
        set_cmo_stream(stream)
        wait_cmo_stream()
        self.assertIs(self.npu.current.waited_on[0], stream)


class TestPrepareWeightCache(_NPUStreamTestBase):
    @patch("torch_npu.npu_prefetch")
    def test_single_weight_creates_stream_and_prefetches(self, prefetch):
        handle = object()
        weight = torch.randn(2, 2)
        prepare_weight_cache(handle, weight)
        prefetch.assert_called_once_with(weight, handle, DEFAULT_PREFETCH_MAX_SIZE)

        stream = get_cmo_stream()
        self.assertIsNotNone(stream)
        self.assertIn(stream, self.npu.created)  # a new stream was registered
        self.assertEqual(len(stream.waited_on), 1)  # it waited on the current stream
        self.assertIs(stream.waited_on[0], self.npu.current)

    @patch("torch_npu.npu_prefetch")
    def test_list_cache_prefetches_each_weight(self, prefetch):
        handle = object()
        w1 = torch.randn(2, 2)
        w2 = torch.randn(3, 3)
        prepare_weight_cache(handle, [w1, w2])
        self.assertEqual(prefetch.call_count, 2)
        # Identity check (== on differently-shaped tensors raises).
        self.assertIs(prefetch.call_args_list[0].args[0], w1)
        self.assertIs(prefetch.call_args_list[1].args[0], w2)
        for call in prefetch.call_args_list:
            self.assertIs(call.args[1], handle)
            self.assertEqual(call.args[2], DEFAULT_PREFETCH_MAX_SIZE)

    @patch("torch_npu.npu_prefetch")
    def test_custom_prefetch_max_size(self, prefetch):
        handle = object()
        weight = torch.randn(2, 2)
        prepare_weight_cache(handle, weight, PREFETCH_MAX_SIZE=1234)
        prefetch.assert_called_once_with(weight, handle, 1234)

    @patch("torch_npu.npu_prefetch")
    def test_reuses_existing_stream(self, prefetch):
        handle = object()
        existing = _FakeStream("existing")
        set_cmo_stream(existing)
        prepare_weight_cache(handle, torch.randn(2, 2))
        self.assertIs(get_cmo_stream(), existing)
        self.assertEqual(self.npu.created, [])  # no new stream was created
        self.assertEqual(len(existing.waited_on), 1)
        prefetch.assert_called_once()


class TestShareStream(_NPUStreamTestBase):
    def test_get_share_stream_returns_none_after_reset(self):
        self.assertIsNone(get_share_stream())

    def test_set_and_get_share_stream(self):
        stream = _FakeStream("share")
        set_share_stream(stream)
        self.assertIs(get_share_stream(), stream)

    def test_wait_share_stream_noop_when_unset(self):
        wait_share_stream()
        self.assertEqual(self.npu.current.waited_on, [])

    def test_wait_share_stream_makes_current_wait(self):
        stream = _FakeStream("share")
        set_share_stream(stream)
        wait_share_stream()
        self.assertIs(self.npu.current.waited_on[0], stream)

    def test_shared_expert_runs_forward_on_new_stream(self):
        hidden = torch.randn(4, 8)
        ran_on = {}

        def forward(x):
            ran_on["stream"] = self.npu.current
            return x * 2

        out = shared_expert_on_independent_stream(hidden, forward)
        self.assertTrue(torch.equal(out, hidden * 2))
        # the forward ran inside the newly created share stream...
        self.assertIs(ran_on["stream"], get_share_stream())
        # ...which waited on the caller's current stream...
        self.assertEqual(len(ran_on["stream"].waited_on), 1)
        self.assertIs(ran_on["stream"].waited_on[0], self.npu.current)
        # ...and the caller's current stream is restored afterwards.
        self.assertIsNot(self.npu.current, ran_on["stream"])

    def test_shared_expert_reuses_existing_stream(self):
        existing = _FakeStream("existing")
        set_share_stream(existing)
        ran_on = {}

        def forward(x):
            ran_on["stream"] = self.npu.current
            return x

        shared_expert_on_independent_stream(torch.randn(2, 2), forward)
        self.assertIs(ran_on["stream"], existing)
        self.assertEqual(self.npu.created, [])  # no new stream was created


if __name__ == "__main__":
    unittest.main()
