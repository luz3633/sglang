"""
Unit tests for sglang.srt.hardware_backend.npu.extra_ops_loader.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
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

# torch_npu is a pre-load import of the DSpark spec; provide a stub so the
# initialize_dspark_sparse_attn_ops entry point is importable on a CPU-only box.
sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.extra_ops_loader import (  # noqa: E402
    OpLibSpec,
    TorchOpLoader,
    initialize_dspark_sparse_attn_ops,
)

NS = "_unit_test_extra_ops"
OPS = ("op_a", "op_b")
SO_ENV = "SGLANG_UT_EXTRA_OPS_SO"


def _make_spec(env=SO_ENV, namespace=NS, ops=OPS, pre_load_imports=()):
    return OpLibSpec(
        name="unit-test lib",
        so_env=env,
        namespace=namespace,
        required_ops=ops,
        pre_load_imports=pre_load_imports,
    )


def _make_ops(namespace, ops, with_load_library=True):
    attrs = {namespace: types.SimpleNamespace(**{op: object() for op in ops})}
    if with_load_library:
        attrs["load_library"] = MagicMock()
    return types.SimpleNamespace(**attrs)


def _register_ops_side_effect(fake_ops, namespace, ops):
    def _side_effect(path):
        setattr(fake_ops, namespace, types.SimpleNamespace(**{op: object() for op in ops}))

    return _side_effect


class _TempSoMixin:
    def _temp_so(self):
        fd, path = tempfile.mkstemp(suffix=".so")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return Path(path)


class TestOpLibSpec(unittest.TestCase):
    def test_fields(self):
        spec = _make_spec()
        self.assertEqual(spec.name, "unit-test lib")
        self.assertEqual(spec.so_env, SO_ENV)
        self.assertEqual(spec.namespace, NS)
        self.assertEqual(spec.required_ops, OPS)

    def test_default_pre_load_imports(self):
        spec = OpLibSpec(name="x", so_env="E", namespace="n", required_ops=("a",))
        self.assertEqual(spec.pre_load_imports, ())


class TestTorchOpLoader(_TempSoMixin, unittest.TestCase):
    def test_missing_ops_returns_all_when_namespace_absent(self):
        with patch.object(torch, "ops", _make_ops("_other_ns", ())):
            loader = TorchOpLoader(_make_spec())
            self.assertEqual(loader._missing_ops(), list(OPS))
            self.assertFalse(loader.registered())

    def test_registered_false_when_some_ops_missing(self):
        with patch.object(torch, "ops", _make_ops(NS, ("op_a",))):
            loader = TorchOpLoader(_make_spec())
            self.assertEqual(loader._missing_ops(), ["op_b"])
            self.assertFalse(loader.registered())

    def test_registered_true_when_all_ops_present(self):
        with patch.object(torch, "ops", _make_ops(NS, OPS)):
            loader = TorchOpLoader(_make_spec())
            self.assertTrue(loader.registered())
            self.assertEqual(loader._missing_ops(), [])

    def test_resolve_so_path_raises_when_env_unset(self):
        loader = TorchOpLoader(_make_spec(env="SGLANG_UT_MISSING_SO"))
        with self.assertRaises(RuntimeError) as ctx:
            loader._resolve_so_path()
        self.assertIn("SGLANG_UT_MISSING_SO", str(ctx.exception))

    def test_resolve_so_path_raises_when_file_missing(self):
        loader = TorchOpLoader(_make_spec(env="SGLANG_UT_MISSING_SO"))
        with patch.dict(
            os.environ, {"SGLANG_UT_MISSING_SO": str(Path.home() / "no_such_file.so")}
        ):
            with self.assertRaises(RuntimeError) as ctx:
                loader._resolve_so_path()
        self.assertIn("missing file", str(ctx.exception))

    def test_resolve_so_path_returns_resolved_path(self):
        so = self._temp_so()
        loader = TorchOpLoader(_make_spec(env="SGLANG_UT_RESOLVE_SO"))
        with patch.dict(os.environ, {"SGLANG_UT_RESOLVE_SO": str(so)}):
            self.assertEqual(loader._resolve_so_path(), so.resolve())

    def test_validate_python_abi_matching(self):
        loader = TorchOpLoader(_make_spec())
        abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        loader._validate_python_abi(Path(f"libfoo.{abi}-x86_64-linux-gnu.so"))  # no raise

    def test_validate_python_abi_mismatch(self):
        loader = TorchOpLoader(_make_spec())
        current = f"{sys.version_info.major}{sys.version_info.minor}"
        other = "0" if current != "0" else "1"
        with self.assertRaises(RuntimeError) as ctx:
            loader._validate_python_abi(Path(f"libfoo.cpython-{other}-x86_64-linux-gnu.so"))
        self.assertIn(f"cpython-{other}", str(ctx.exception))

    def test_validate_python_abi_ignores_unversioned_name(self):
        loader = TorchOpLoader(_make_spec())
        loader._validate_python_abi(Path("libfoo.so"))  # no raise

    def test_initialize_returns_none_when_registered(self):
        so = self._temp_so()
        fake_ops = _make_ops(NS, OPS)
        with patch.object(torch, "ops", fake_ops), patch.dict(
            os.environ, {SO_ENV: str(so)}
        ):
            loader = TorchOpLoader(_make_spec())
            self.assertIsNone(loader.initialize())
        fake_ops.load_library.assert_not_called()

    def test_initialize_loads_library_and_returns_path(self):
        so = self._temp_so()
        fake_ops = _make_ops(NS, ())
        fake_ops.load_library.side_effect = _register_ops_side_effect(fake_ops, NS, OPS)
        with patch.object(torch, "ops", fake_ops), patch.dict(
            os.environ, {SO_ENV: str(so)}
        ):
            loader = TorchOpLoader(_make_spec())
            result = loader.initialize()
        self.assertEqual(result, so.resolve())
        self.assertEqual(loader._loaded_library, so.resolve())
        fake_ops.load_library.assert_called_once_with(str(so.resolve()))

    def test_initialize_raises_when_ops_still_missing_after_load(self):
        so = self._temp_so()
        fake_ops = _make_ops(NS, ())  # load_library has no side effect
        with patch.object(torch, "ops", fake_ops), patch.dict(
            os.environ, {SO_ENV: str(so)}
        ):
            loader = TorchOpLoader(_make_spec())
            with self.assertRaises(RuntimeError) as ctx:
                loader.initialize()
        self.assertIn("operators are missing", str(ctx.exception))

    def test_initialize_raises_when_already_loaded_but_missing(self):
        loader = TorchOpLoader(_make_spec())
        loader._loaded_library = Path("/fake/loaded.so")
        with patch.object(torch, "ops", _make_ops(NS, ())):
            with self.assertRaises(RuntimeError) as ctx:
                loader.initialize()
        self.assertIn("Loaded", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))

    def test_initialize_raises_on_load_library_failure(self):
        so = self._temp_so()
        fake_ops = _make_ops(NS, ())
        fake_ops.load_library.side_effect = OSError("boom")
        with patch.object(torch, "ops", fake_ops), patch.dict(
            os.environ, {SO_ENV: str(so)}
        ):
            loader = TorchOpLoader(_make_spec())
            with self.assertRaises(RuntimeError) as ctx:
                loader.initialize()
        self.assertIn("Failed to load", str(ctx.exception))

    def test_initialize_raises_when_so_env_unset(self):
        fake_ops = _make_ops(NS, ())
        with patch.object(torch, "ops", fake_ops), patch.dict(os.environ, {}, clear=True):
            loader = TorchOpLoader(_make_spec())
            with self.assertRaises(RuntimeError):
                loader.initialize()

    @patch("builtins.__import__", wraps=__import__)
    def test_initialize_imports_pre_load_modules(self, import_mock):
        so = self._temp_so()
        fake_ops = _make_ops(NS, ())
        fake_ops.load_library.side_effect = _register_ops_side_effect(fake_ops, NS, OPS)
        with patch.object(torch, "ops", fake_ops), patch.dict(
            os.environ, {SO_ENV: str(so)}
        ):
            loader = TorchOpLoader(_make_spec(pre_load_imports=("torch_npu",)))
            loader.initialize()
        names = [c.args[0] for c in import_mock.call_args_list]
        self.assertIn("torch_npu", names)


class TestInitializeDSparkSparseAttnOps(_TempSoMixin, unittest.TestCase):
    def test_registers_dspark_ops(self):
        so = self._temp_so()
        ns = "_C_ascend"
        ops = ("npu_sparse_attn_sharedkv_metadata", "npu_sparse_attn_sharedkv")
        fake_ops = _make_ops(ns, ())
        fake_ops.load_library.side_effect = _register_ops_side_effect(fake_ops, ns, ops)
        with patch.object(torch, "ops", fake_ops), patch.dict(
            os.environ, {"SGLANG_DSPARK_EXTRA_OPS_SO": str(so)}
        ):
            result = initialize_dspark_sparse_attn_ops()
        self.assertEqual(result, so.resolve())
        fake_ops.load_library.assert_called_once_with(str(so.resolve()))


if __name__ == "__main__":
    unittest.main()
