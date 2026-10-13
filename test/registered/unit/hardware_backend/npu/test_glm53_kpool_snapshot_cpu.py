"""CPU-only copy/verify contract tests; not an NPU compilation or speed claim."""

import ast
from copy import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/glm53"
spec = importlib.util.spec_from_file_location("snapshot_cpu", SOURCE / "kpool_snapshot_npu.py")
kernel = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
    spec.loader.exec_module(kernel)


def reference(keys, scores, requests, snapshot_keys, snapshot_scores, step):
    n = requests.numel()
    snapshot_keys[:n, step].copy_(keys.index_select(0, requests.long()))
    snapshot_scores[:n, step].copy_(scores.index_select(0, requests.long()))


def buffers(batch=2, width=128):
    k = torch.randn(9, 4, width, dtype=torch.bfloat16)
    s = torch.randn(9, 4, width, dtype=torch.float32)
    sk = torch.randn(batch + 2, 4, 4, width, dtype=torch.bfloat16)
    ss = torch.randn(batch + 2, 4, 4, width)
    return k, s, sk, ss


class SnapshotCPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        env = patch.dict(os.environ, {"TRITON_INTERPRET": "1"})
        env.start()
        self.addCleanup(env.stop)

    def bits(self, a, b):
        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def test_shapes_steps_padding_duplicate_reads_and_dynamic_indices(self):
        for n in (0, 1, 2, 4, 8):
            for width in (128, 129):
                for dtype in (torch.int32, torch.int64):
                    k, s, sk, ss = buffers(n, width)
                    ek, es = sk.clone(), ss.clone()
                    requests = torch.zeros(n, dtype=dtype)
                    addresses = sk.data_ptr(), ss.data_ptr()
                    for iteration in range(3):
                        requests.copy_((torch.arange(n) + iteration) % 9)
                        if n >= 2:
                            requests[-1] = requests[0]  # Duplicate is a safe read.
                        for step in range(4):
                            reference(k, s, requests, ek, es, step)
                            kernel.snapshot_kpool_tails_(k, s, requests, sk, ss, step)
                            self.bits(sk, ek)
                            self.bits(ss, es)
                            self.assertEqual((sk.data_ptr(), ss.data_ptr()), addresses)

    def test_special_bit_patterns_and_unselected_steps(self):
        k = torch.tensor([0, -32768, 1, 32640, -128, 32705, 32767, -1], dtype=torch.int16).view(torch.bfloat16).repeat(9, 4, 1)
        s = torch.tensor([0, -2147483648, 1, 2139095040, -8388608, 2143289345, 2147483647, -1], dtype=torch.int32).view(torch.float32).repeat(9, 4, 1)
        sk, ss = torch.ones(4, 4, 4, 8, dtype=k.dtype), torch.ones(4, 4, 4, 8)
        ek, es = sk.clone(), ss.clone()
        requests = torch.tensor([8, 0])
        reference(k, s, requests, ek, es, 2)
        kernel.snapshot_kpool_tails_(k, s, requests, sk, ss, 2)
        self.bits(sk, ek); self.bits(ss, es)

    def test_real_verify_method_preserves_outputs_snapshots_and_rollback_selection(self):
        tree = ast.parse((SOURCE / "kpool_indexer.py").read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendIndexerKPoolMixin")
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_topk")
        namespace = dict(torch=torch, copy=copy, _get_full_attn_metadata=lambda b: b._ascend_kpool_metadata, snapshot_kpool_tails_=kernel.snapshot_kpool_tails_)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "verify_isolated", "exec"), namespace)
        for n in (1, 2):
            k, s, sk, ss = buffers(n)
            instances = []
            for enabled in (False, True):
                obj = NS(_kpool_tail_k=k.clone(), _kpool_tail_score=s.clone(), _kpool_mtp_tail_k=sk.clone(), _kpool_mtp_tail_score=ss.clone(), _fuse_kpool_verify_snapshot=enabled)
                def decode(q, key, weights, gate, pos, batch, tables, layer, owner=obj):
                    for row, req in enumerate(batch.req_pool_indices.tolist()):
                        slot = int(pos[row]) % 4
                        owner._kpool_tail_k[req, slot].copy_(key[row])
                        owner._kpool_tail_score[req, slot].copy_(gate[row])
                    return pos[:, None].expand(-1, 7).int().clone()
                obj._decode_topk = decode
                instances.append(obj)
            batch = NS(req_pool_indices=torch.tensor([7, 0][:n]), batch_size=n, _original_batch_size=n,
                       spec_info=NS(draft_token_num=4), _ascend_kpool_metadata=NS(seq_lens=torch.zeros(n,dtype=torch.int32)))
            q = torch.randn(n*4+2, 2, 128)
            key = torch.randn(n*4+2, 128, dtype=torch.bfloat16)
            weights = torch.randn(n*4+2, 2)
            gate = torch.randn(n*4+2, 128)
            positions = torch.arange(n*4+2) + 262140
            outputs = [namespace['_verify_topk'](obj, q, key, weights, gate, positions, batch, torch.zeros(n, 5, dtype=torch.int32), 3) for obj in instances]
            self.bits(*outputs)
            for attr in ('_kpool_tail_k', '_kpool_tail_score', '_kpool_mtp_tail_k', '_kpool_mtp_tail_score'):
                self.bits(getattr(instances[0], attr), getattr(instances[1], attr))
            for accepted in range(4):
                for attr in ('_kpool_mtp_tail_k', '_kpool_mtp_tail_score'):
                    self.bits(getattr(instances[0], attr)[:n, accepted], getattr(instances[1], attr)[:n, accepted])

    def test_invalid_layout_fails_before_launch(self):
        k, s, sk, ss = buffers()
        for requests, step in ((torch.tensor([1.]),0), (torch.tensor([1]),-1), (torch.tensor([1]),4), (torch.tensor([[1]]),0)):
            with self.assertRaises(ValueError):
                kernel.snapshot_kpool_tails_(k, s, requests, sk, ss, step)
        with self.assertRaises(ValueError):
            kernel.snapshot_kpool_tails_(k[:,:,::2], s[:,:,::2], torch.tensor([1]), sk[:,:,:,::2], ss[:,:,:,::2], 0)


if __name__ == '__main__':
    unittest.main()
