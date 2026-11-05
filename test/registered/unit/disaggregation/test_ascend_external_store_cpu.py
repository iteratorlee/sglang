"""CPU-only contract for attaching new P replicas to an existing Ascend store."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class TestAscendExternalStore(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        source = root / "python/sglang/srt/managers/disagg_service.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "maybe_create_ascend_config_store")
        self.create = Mock()
        self.parallel = SimpleNamespace(node_rank=0)
        namespace = {"os": os, "TransferBackend": SimpleNamespace(ASCEND="ascend"), "get_parallel": lambda: self.parallel}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.function = namespace[function.name]

    def invoke(self, env, backend="ascend"):
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"memfabric_hybrid": SimpleNamespace(create_config_store=self.create)}):
            return self.function(backend)

    def test_original_owner_keeps_creating_store(self):
        self.invoke({"ASCEND_MF_STORE_URL": "tcp://127.0.0.1:26126"})
        self.create.assert_called_once_with("tcp://127.0.0.1:26126")

    def test_new_replica_joins_without_creating_remote_store(self):
        self.invoke({"ASCEND_MF_STORE_URL": "tcp://10.0.0.29:26126", "SGLANG_ASCEND_MF_REUSE_CONFIG_STORE": "1"})
        self.create.assert_not_called()

    def test_missing_external_address_is_a_configuration_error(self):
        with self.assertRaisesRegex(ValueError, "ASCEND_MF_STORE_URL"):
            self.invoke({"SGLANG_ASCEND_MF_REUSE_CONFIG_STORE": "1"})
        self.create.assert_not_called()

    def test_other_ranks_and_backends_do_not_own_store(self):
        self.invoke({}, backend="mooncake")
        self.parallel.node_rank = 1
        self.invoke({"SGLANG_ASCEND_MF_REUSE_CONFIG_STORE": "1"})
        self.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
