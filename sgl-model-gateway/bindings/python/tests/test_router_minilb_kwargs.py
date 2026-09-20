"""Regression tests for keeping MiniLB-only args out of the Rust Router.

No test starts a router or opens a socket.  The offline tests import the real
Python mapping with a strict stub for the native module, so they run even when
the PyO3 extension is unavailable.  The optional native test constructs the
real Rust Router without calling ``start()`` when its signature matches the
source, and reports a source/binary baseline mismatch as a factual skip.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


BINDINGS = Path(__file__).resolve().parents[1]
SOURCE = BINDINGS / "src" / "sglang_router"
NEW_MINI_LB_ONLY_FIELDS = {
    "mini_lb_prefix_affinity_decode_capacity",
    "mini_lb_session_affinity",
    "mini_lb_session_affinity_idle_timeout_secs",
}
MINI_LB_ONLY_FIELDS = {
    "mini_lb",
    "test_external_dp_routing",
    "mini_lb_prefix_affinity",
    "mini_lb_prefix_affinity_length",
    *NEW_MINI_LB_ONLY_FIELDS,
}
UNEXPECTED_KEYWORD_PREFIX = "got an unexpected keyword argument "


def unsupported_baseline_keyword(exc, args):
    """Return an unsupported source-baseline field without hiding MiniLB leaks."""

    message = str(exc)
    if UNEXPECTED_KEYWORD_PREFIX not in message:
        return None
    keyword = message.rsplit(UNEXPECTED_KEYWORD_PREFIX, 1)[1].strip(" '\".")
    if keyword in MINI_LB_ONLY_FIELDS:
        raise AssertionError(f"MiniLB-only kwarg reached native Router: {keyword}")
    return keyword if keyword in vars(args) else None


def _enum_type(name, members):
    enum_type = type(name, (), {})
    for member in members:
        setattr(enum_type, member, enum_type())
    return enum_type


def load_with_strict_native_stub():
    """Load the production RouterArgs/from_args path without native deps."""

    module_names = (
        "sglang_router",
        "sglang_router.router_args",
        "sglang_router.router",
        "sglang_router.sglang_router_rs",
    )
    saved = {name: sys.modules.get(name) for name in module_names}
    captured = {}

    class StrictRouter:
        def __init__(self, **kwargs):
            leaked = MINI_LB_ONLY_FIELDS & set(kwargs)
            if leaked:
                raise TypeError(
                    f"MiniLB-only kwargs reached native Router: {sorted(leaked)}"
                )
            captured.update(kwargs)

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    try:
        package = types.ModuleType("sglang_router")
        package.__path__ = [str(SOURCE)]
        native = types.ModuleType("sglang_router.sglang_router_rs")
        native.PolicyType = _enum_type(
            "PolicyType",
            (
                "Random",
                "RoundRobin",
                "CacheAware",
                "PowerOfTwo",
                "Bucket",
                "Manual",
                "ConsistentHashing",
                "PrefixHash",
            ),
        )
        native.BackendType = _enum_type("BackendType", ("Sglang", "Openai"))
        native.HistoryBackendType = _enum_type(
            "HistoryBackendType", ("Memory", "None", "Oracle", "Postgres", "Redis")
        )
        native.PyRole = _enum_type("PyRole", ("Admin", "User"))
        for name in (
            "PyApiKeyEntry",
            "PyControlPlaneAuthConfig",
            "PyJwtConfig",
            "PyOracleConfig",
            "PyPostgresConfig",
            "PyRedisConfig",
        ):
            setattr(native, name, Config)
        native.Router = StrictRouter
        native.get_available_tool_call_parsers = lambda: []
        sys.modules["sglang_router"] = package
        sys.modules["sglang_router.sglang_router_rs"] = native

        args_spec = importlib.util.spec_from_file_location(
            "sglang_router.router_args", SOURCE / "router_args.py"
        )
        if args_spec is None or args_spec.loader is None:  # pragma: no cover
            raise RuntimeError("cannot load router_args.py")
        args_module = importlib.util.module_from_spec(args_spec)
        sys.modules[args_spec.name] = args_module
        args_spec.loader.exec_module(args_module)

        router_spec = importlib.util.spec_from_file_location(
            "sglang_router.router", SOURCE / "router.py"
        )
        if router_spec is None or router_spec.loader is None:  # pragma: no cover
            raise RuntimeError("cannot load router.py")
        router_module = importlib.util.module_from_spec(router_spec)
        sys.modules[router_spec.name] = router_module
        router_spec.loader.exec_module(router_module)
        return router_module, args_module.RouterArgs, captured
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class OfflineMappingTests(unittest.TestCase):
    def test_default_regular_router_does_not_forward_minilb_fields(self):
        router_module, RouterArgs, captured = load_with_strict_native_stub()
        args = RouterArgs(
            worker_urls=["http://worker:8000"],
            policy="round_robin",
        )

        wrapped = router_module.Router.from_args(args)

        self.assertIsNotNone(wrapped._router)
        self.assertFalse(MINI_LB_ONLY_FIELDS & set(captured))
        self.assertEqual(captured["worker_urls"], ["http://worker:8000"])
        self.assertIs(captured["policy"], router_module.PolicyType.RoundRobin)

    def test_nondefault_minilb_values_are_python_only(self):
        router_module, RouterArgs, captured = load_with_strict_native_stub()
        args = RouterArgs(
            worker_urls=["http://worker:8000"],
            policy="random",
            mini_lb_prefix_affinity_decode_capacity=2,
            mini_lb_session_affinity=True,
            mini_lb_session_affinity_idle_timeout_secs=123.0,
        )

        router_module.Router.from_args(args)

        self.assertFalse(MINI_LB_ONLY_FIELDS & set(captured))

    def test_native_baseline_mismatch_never_hides_new_minilb_leaks(self):
        baseline_args = types.SimpleNamespace(json_log=False)
        baseline_error = TypeError(
            "Router.__new__() got an unexpected keyword argument 'json_log'"
        )
        self.assertEqual(
            unsupported_baseline_keyword(baseline_error, baseline_args), "json_log"
        )

        for field in NEW_MINI_LB_ONLY_FIELDS:
            with self.subTest(field=field), self.assertRaisesRegex(
                AssertionError, field
            ):
                unsupported_baseline_keyword(
                    TypeError(
                        "Router.__new__() got an unexpected keyword argument "
                        f"'{field}'"
                    ),
                    types.SimpleNamespace(**{field: object()}),
                )


NATIVE_IMPORT_ERROR = None
try:
    source_path = str(BINDINGS / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from sglang_router.router import Router as NativeMappedRouter
    from sglang_router.router_args import RouterArgs as NativeRouterArgs
    from sglang_router.sglang_router_rs import Router as NativeRouter
except (ImportError, OSError) as exc:  # pragma: no cover - environment specific
    NATIVE_IMPORT_ERROR = exc
    NativeMappedRouter = NativeRouterArgs = NativeRouter = None


@unittest.skipIf(
    NATIVE_IMPORT_ERROR is not None,
    f"native router extension unavailable: {NATIVE_IMPORT_ERROR}",
)
class NativeConstructorTests(unittest.TestCase):
    def test_default_regular_router_constructs_without_network_or_start(self):
        args = NativeRouterArgs(
            worker_urls=["http://127.0.0.1:65535"],
            policy="round_robin",
        )

        try:
            wrapped = NativeMappedRouter.from_args(args)
        except TypeError as exc:
            unsupported = unsupported_baseline_keyword(exc, args)
            if unsupported is not None:
                self.skipTest(
                    "loaded native extension is incompatible with the source "
                    f"Router signature: existing baseline field {unsupported!r} "
                    f"is unsupported ({exc})"
                )
            raise

        self.assertIsInstance(wrapped._router, NativeRouter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
