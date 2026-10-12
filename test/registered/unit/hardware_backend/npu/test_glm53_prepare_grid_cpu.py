"""Exercise actual bounded-grid row addressing without importing a device runtime."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[5]/(
    'python/sglang/srt/hardware_backend/npu/attention/glm53/kda_prefill_npu.py')


def addressing():
    tree = ast.parse(SOURCE.read_text())
    kernel = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_prepare_inputs')
    loop = copy.deepcopy(next(n for n in kernel.body if isinstance(n,ast.For)))
    guard = next(n for n in loop.body if isinstance(n,ast.If))
    # Retain production addressing and its bounds guard; replace tensor math
    # with observation of the rows that would actually be read/written.
    guard.body = ast.parse('seen.append(row)').body
    fn = ast.parse('def rows(tl, N_ROWS):\n seen=[]\n return seen').body[0]
    fn.body.insert(1,loop)
    module = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace = {}
    exec(compile(module,str(SOURCE),'exec'),namespace)
    return namespace['rows']


def production_grid(rows):
    tree = ast.parse(SOURCE.read_text())
    wrapper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'run_prepared_prefill')
    call = next(n for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)
                and isinstance(n.func.value,ast.Name) and n.func.value.id == '_prepare_inputs')
    return eval(compile(ast.Expression(call.func.slice),str(SOURCE),'eval'),{'rows':rows})[0]


class PrepareGridCPU(unittest.TestCase):
    def test_all_rows_exactly_once_including_launch_limit_and_partial_tail(self):
        visit = addressing()
        for rows in (1,128*4,4096*8,32769,65535,8192*8,8193*8,16384*8):
            with self.subTest(rows=rows):
                grid = production_grid(rows)
                self.assertGreater(grid,0)
                self.assertLessEqual(grid,32768)
                seen = []
                for pid in range(grid):
                    tl = SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,
                        program_id=lambda dim,pid=pid:pid,num_programs=lambda dim:grid)
                    seen.extend(visit(tl,rows))
                self.assertEqual(sorted(seen),list(range(rows)))
                self.assertEqual(len(seen),len(set(seen)))


if __name__ == '__main__':
    unittest.main()
