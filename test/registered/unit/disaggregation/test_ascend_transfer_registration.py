"""Registration failures must stop Ascend PD before a KV transfer begins."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def load_batch_register():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/disaggregation/ascend/transfer_engine.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendTransferEngine"
    )
    method = copy.deepcopy(
        next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "batch_register"
        )
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"List": list}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["batch_register"]


class AscendRegistrationTests(unittest.TestCase):
    def test_success(self):
        engine = Mock()
        engine.batch_register_memory.return_value = 0
        load_batch_register()(SimpleNamespace(engine=engine), [123], [456])
        engine.batch_register_memory.assert_called_once_with([123], [456])

    def test_nonzero_result_fails(self):
        engine = Mock()
        engine.batch_register_memory.return_value = -8
        with self.assertRaisesRegex(RuntimeError, "return code -8"):
            load_batch_register()(SimpleNamespace(engine=engine), [123], [456])

    def test_exception_fails_with_cause(self):
        engine = Mock()
        engine.batch_register_memory.side_effect = OSError("registration failed")
        with self.assertRaisesRegex(RuntimeError, "1 buffers") as caught:
            load_batch_register()(SimpleNamespace(engine=engine), [123], [456])
        self.assertIsInstance(caught.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
