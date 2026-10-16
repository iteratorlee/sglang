"""P draft scope consensus, fabricated idle mode and lifecycle on CPU only."""
import ast
import os
from abc import ABC
from contextlib import contextmanager
from enum import IntEnum, auto
import sys
from types import ModuleType
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import test_glm53_normal_idle_dp_cpu as idle
from test_glm53_normal_hccl_cpu import SRT, load_source


class DraftPrefillScopeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # Execute the actual production phase definitions, rather than a fake
        # predicate: EAGLE_DRAFT_EXTEND.is_draft_input() is deliberately True.
        parsed = ast.parse((SRT / 'speculative/spec_info.py').read_text())
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in ('SpecInputType', 'SpecInput')]
        namespace = dict(ABC=ABC, IntEnum=IntEnum, auto=auto)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<native spec phases>', 'exec'), namespace)
        self.SpecInputType, self.SpecInput = namespace['SpecInputType'], namespace['SpecInput']
        spec_module = ModuleType('sglang.srt.speculative.spec_info')
        spec_module.SpecInputType = self.SpecInputType
        mock.patch.dict(sys.modules, {spec_module.__name__: spec_module}).start()
        self.helper = load_source('draft_prefill_helper', SRT / 'hardware_backend/npu/moe/glm53_draft_prefill.py')

    def batch(self, counts, rank):
        self.fixture.parallel.attn_dp_rank = rank
        batch = idle.batch(counts)
        batch.is_extend_in_batch = True
        batch.spec_info = self.SpecInput(self.SpecInputType.EAGLE_DRAFT_EXTEND)
        batch.spec_info.num_tokens_per_req = 1
        self.assertTrue(batch.spec_info.is_draft_input())
        return batch

    def test_both_domains_publish_identical_original_counts(self):
        for counts in ((16384, 0), (0, 16384), (16384, 7), (7, 16384), (256, 256)):
            for rank in (0, 1):
                with self.subTest(counts=counts, rank=rank):
                    batch = self.batch(counts, rank)
                    old = vars(batch).copy()
                    with self.helper.draft_prefill_scope(batch):
                        self.assertEqual(self.fixture.module._idle_dp_counts.get(), counts)
                        self.assertEqual(vars(batch), old)
                    self.assertIsNone(self.fixture.module._idle_dp_counts.get())

    def test_real_decode_phases_do_not_publish_counts(self):
        for mode in idle.Mode:
            batch = self.batch((32, 0), 0)
            batch.is_extend_in_batch = False
            batch.forward_mode = mode
            with self.helper.draft_prefill_scope(batch):
                self.assertIsNone(self.fixture.module._idle_dp_counts.get())

    def test_rejects_real_verify_and_non_prefill_draft_geometry(self):
        batch = self.batch((32, 0), 0)
        batch.forward_mode = idle.Mode.TARGET_VERIFY
        with self.assertRaisesRegex(ValueError, 'native EXTEND'):
            with self.helper.draft_prefill_scope(batch):
                self.fail('invalid mode admitted')
        batch = self.batch((32, 0), 0)
        batch.spec_info.num_tokens_per_req = 4
        with self.assertRaisesRegex(ValueError, 'one-token-per-row'):
            with self.helper.draft_prefill_scope(batch):
                self.fail('decode draft geometry admitted')
        for kind in (self.SpecInputType.EAGLE_DRAFT, self.SpecInputType.EAGLE_VERIFY):
            batch = self.batch((32, 0), 0)
            batch.spec_info.spec_input_type = kind
            with self.assertRaisesRegex(ValueError, 'EAGLE_DRAFT_EXTEND'):
                with self.helper.draft_prefill_scope(batch):
                    self.fail('wrong phase admitted')

    def test_context_restores_on_success_and_error(self):
        events = []
        @contextmanager
        def context():
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with self.helper.draft_prefill_scope(self.batch((32, 0), 1), context()):
                raise RuntimeError('test')
        self.assertEqual(events, ['enter', 'exit'])
        self.assertIsNone(self.fixture.module._idle_dp_counts.get())

    def test_shared_decoder_defaults_off_and_glm_constructor_admits(self):
        source = ast.parse((SRT / 'models/deepseek_v2.py').read_text())
        decoder = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == 'DeepseekV2DecoderLayer')
        init = next(x for x in decoder.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
        assignments = [x for x in ast.walk(init) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == '_glm53_normal_prefill_draft' for t in x.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertIs(assignments[0].value.value, False)
        forward = next(x for x in decoder.body if isinstance(x, ast.FunctionDef) and x.name == 'forward')
        self.assertIn('draft_prefill_scope(forward_batch, _mlp_ctx)', ast.unparse(forward))
        glm = (SRT / 'models/glm5_next_nextn.py').read_text()
        self.assertIn('SGLANG_GLM53_NORMAL_PREFILL_DRAFT_SCOPE', glm)
        self.assertIn('normal_hccl_skip_idle_dp_enabled()', glm)

    def test_minimum_uses_original_global_vector_on_active_and_idle_peers(self):
        with mock.patch.dict(os.environ, {'SGLANG_GLM53_NORMAL_PREFILL_DRAFT_MIN_TOKENS': '4096'}):
            for counts, admitted in [((256, 0), False), ((0, 4095), False),
                                     ((0, 4096), True), ((16384, 7), True),
                                     ((7, 16384), True), ((2048, 2048), False)]:
                for rank in (0, 1):
                    with self.subTest(counts=counts, rank=rank):
                        batch = self.batch(counts, rank)
                        before = vars(batch).copy()
                        with self.helper.draft_prefill_scope(batch):
                            self.assertEqual(self.fixture.module._idle_dp_counts.get(), counts if admitted else None)
                            self.assertEqual(vars(batch), before)
                        self.assertIsNone(self.fixture.module._idle_dp_counts.get())

    def test_small_fallback_preserves_original_mlp_context_on_error(self):
        events = []
        @contextmanager
        def context():
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')
        with mock.patch.dict(os.environ, {'SGLANG_GLM53_NORMAL_PREFILL_DRAFT_MIN_TOKENS': '4096'}):
            with self.assertRaisesRegex(RuntimeError, 'body'):
                with self.helper.draft_prefill_scope(self.batch((128, 0), 1), context()):
                    self.assertIsNone(self.fixture.module._idle_dp_counts.get())
                    raise RuntimeError('body')
        self.assertEqual(events, ['enter', 'exit'])

    def test_negative_minimum_rejected(self):
        with mock.patch.dict(os.environ, {'SGLANG_GLM53_NORMAL_PREFILL_DRAFT_MIN_TOKENS': '-1'}):
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                with self.helper.draft_prefill_scope(self.batch((128, 0), 0)):
                    self.fail('invalid minimum admitted')

    def test_padded_4096_does_not_admit_original_4095(self):
        with mock.patch.dict(os.environ, {'SGLANG_GLM53_NORMAL_PREFILL_DRAFT_MIN_TOKENS': '4096'}):
            for rank in (0, 1):
                batch = self.batch((0, 4095), rank)
                batch.global_num_tokens_cpu = [4096, 4096]
                with self.helper.draft_prefill_scope(batch):
                    self.assertIsNone(self.fixture.module._idle_dp_counts.get())


if __name__ == '__main__':
    unittest.main()
