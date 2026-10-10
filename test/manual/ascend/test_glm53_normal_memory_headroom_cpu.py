"""CPU contracts for the real GLM normal HCCL memory-headroom helper."""

import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/moe/glm53_collectives.py"
)
FLAG = "SGLANG_GLM53_NORMAL_MEMORY_HEADROOM_MB"
MIB = 1024**2


def load_production_helper(npu, logger):
    tree = ast.parse(SOURCE.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "release_normal_hccl_cached_memory"
    )
    namespace = {
        "os": os,
        "torch": types.SimpleNamespace(npu=npu),
        "logger": logger,
    }
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[helper.name]


class FakeNPU:
    """Record driver calls and model only the unused allocator cache."""

    def __init__(self, free_mb, reserved_mb, allocated_mb, after_free_mb=0):
        self.free = free_mb * MIB
        self.reserved = reserved_mb * MIB
        self.allocated = allocated_mb * MIB
        self.after_free = after_free_mb * MIB
        self.calls = []

    def mem_get_info(self, device):
        self.calls.append(("mem_get_info", device))
        return self.free, 4096 * MIB

    def memory_reserved(self, device):
        self.calls.append(("memory_reserved", device))
        return self.reserved

    def memory_allocated(self, device):
        self.calls.append(("memory_allocated", device))
        return self.allocated

    def synchronize(self, device):
        self.calls.append(("synchronize", device))

    def empty_cache(self):
        self.calls.append(("empty_cache",))
        self.reserved = self.allocated
        self.free = self.after_free


class NormalMemoryHeadroomCPU(unittest.TestCase):
    def setUp(self):
        self.device = types.SimpleNamespace(type="npu")
        self.x = types.SimpleNamespace(device=self.device)
        self.logger = Mock()

    def run_helper(self, npu, flag):
        helper = load_production_helper(npu, self.logger)
        with patch.dict(os.environ, {}, clear=True):
            if flag is not None:
                os.environ[FLAG] = flag
            return helper(self.x)

    def test_dispatch_calls_helper_only_under_world_eight_guard(self):
        tree = ast.parse(SOURCE.read_text())
        dispatch = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "collective_dispatch"
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "release_normal_hccl_cached_memory"
        ]
        self.assertEqual(len(calls), 1)
        guards = [
            node
            for node in dispatch.body
            if isinstance(node, ast.If) and calls[0] in ast.walk(node)
        ]
        self.assertEqual(len(guards), 1)
        self.assertEqual(
            ast.dump(guards[0].test),
            ast.dump(ast.parse("world == 8", mode="eval").body),
        )

    def test_default_and_zero_do_not_call_driver(self):
        for flag in (None, "0"):
            with self.subTest(flag=flag):
                npu = FakeNPU(1, 512, 100)
                self.assertIsNone(self.run_helper(npu, flag))
                self.assertEqual(npu.calls, [])
                self.logger.warning.assert_not_called()

    def test_cpu_device_does_not_call_driver(self):
        self.x.device = types.SimpleNamespace(type="cpu")
        npu = FakeNPU(1, 512, 100)
        self.assertIsNone(self.run_helper(npu, "128"))
        self.assertEqual(npu.calls, [])
        self.logger.warning.assert_not_called()

    def test_sufficient_free_memory_does_not_flush(self):
        for free_mb in (128, 129):
            with self.subTest(free_mb=free_mb):
                npu = FakeNPU(free_mb, 512, 100)
                self.assertIsNone(self.run_helper(npu, "128"))
                self.assertEqual(npu.calls, [("mem_get_info", self.device)])
                self.logger.warning.assert_not_called()

    def test_low_free_and_64mb_cached_gap_flushes_without_live_allocation(self):
        for reserved_mb in (164, 180):
            with self.subTest(reserved_mb=reserved_mb):
                npu = FakeNPU(32, reserved_mb, 100, after_free_mb=96)
                self.assertIsNone(self.run_helper(npu, "128"))
                self.assertEqual(
                    npu.calls,
                    [
                        ("mem_get_info", self.device),
                        ("memory_reserved", self.device),
                        ("memory_allocated", self.device),
                        ("synchronize", self.device),
                        ("empty_cache",),
                        ("mem_get_info", self.device),
                    ],
                )
                self.assertEqual(npu.allocated, 100 * MIB)
                self.assertEqual(npu.reserved, npu.allocated)
                self.logger.warning.assert_called_once()
                self.assertEqual(self.logger.warning.call_args.args[1:], (32, 96, reserved_mb, 100, 128))
                self.logger.reset_mock()

    def test_cached_gap_below_64mb_does_not_flush(self):
        npu = FakeNPU(32, 163, 100)
        self.assertIsNone(self.run_helper(npu, "128"))
        self.assertEqual(
            npu.calls,
            [
                ("mem_get_info", self.device),
                ("memory_reserved", self.device),
                ("memory_allocated", self.device),
            ],
        )
        self.assertEqual(npu.reserved, 163 * MIB)
        self.assertEqual(npu.allocated, 100 * MIB)
        self.logger.warning.assert_not_called()

    def test_negative_threshold_fails_closed_before_driver_access(self):
        npu = FakeNPU(1, 512, 100)
        with self.assertRaisesRegex(ValueError, FLAG):
            self.run_helper(npu, "-1")
        self.assertEqual(npu.calls, [])
        self.logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
