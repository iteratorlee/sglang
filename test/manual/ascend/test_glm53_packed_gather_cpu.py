"""Portable CPU contract tests for the production bucketed payload helper.

Run: CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     python -B test/manual/ascend/test_glm53_packed_gather_cpu.py

Loads only the helper file, without importing sglang, Triton, or torch_npu.
These tests establish CPU layout/gate contracts, not NPU kernel correctness.
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

torch.set_num_threads(1)

HELPER = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/moe/glm53_packed_gather.py"
)


def raw_bytes(tensor):
    # Empty/singleton tensors can retain unusual strides despite contiguous().
    return tensor.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


def strided_copy(tensor):
    if tensor.ndim == 1:
        storage = torch.empty(tensor.numel() * 3 + 1, dtype=tensor.dtype, device="cpu")
        view = storage[1::3]
    else:
        rows, columns = tensor.shape
        storage = torch.empty((rows, columns * 2 + 3), dtype=tensor.dtype, device="cpu")
        view = storage[:, 1 : 1 + columns * 2 : 2]
    view.copy_(tensor)
    return view


def payload(rows, rank, ids_dtype, strided=False):
    q = ((torch.arange(rows * 4096, device="cpu") + 37 * rank) % 256).to(torch.int8)
    # Finite, signed zero, infinities, subnormals, quiet/signaling/signed NaNs.
    bits = torch.tensor(
        [0, -2147483648, 0x3F800000, -1082130432, 0x7F800000,
         -8388608, 1, -2147483647, 0x7FC01234, 0x7F800001, -4194303],
        dtype=torch.int32, device="cpu",
    )
    scale = bits[(torch.arange(rows, device="cpu") + rank) % len(bits)].view(torch.float32)
    weights = bits[(torch.arange(rows * 8, device="cpu") + 3 * rank) % len(bits)]
    values = [0, 17, 17, -1, torch.iinfo(ids_dtype).min, torch.iinfo(ids_dtype).max]
    if ids_dtype == torch.int64:
        values += [2**32 + 7, 2**53 + 1, -(2**47) + 23]
    ids = torch.tensor(values, dtype=ids_dtype, device="cpu")
    ids = ids[(torch.arange(rows * 8, device="cpu") + rank) % len(ids)]
    tensors = (q.reshape(rows, 4096), scale, ids.reshape(rows, 8),
               weights.view(torch.float32).reshape(rows, 8))
    return tuple(strided_copy(t) for t in tensors) if strided else tensors


class PackedGatherCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = "_glm53_packed_gather_cpu_test_helper"
        spec = importlib.util.spec_from_file_location(name, HELPER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclass resolves its defining module here.
        spec.loader.exec_module(module)
        cls.helper = module
        cls.addClassCleanup(sys.modules.pop, name, None)

    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(raw_bytes(actual), raw_bytes(expected)))

    def test_all_bucket_layouts_traffic_and_memory_bounds(self):
        for dtype in (torch.int64, torch.int32):
            signatures = {}
            for rows in range(513):
                layout = self.helper.payload_layout(rows, dtype)
                capacity = next(c for c in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                                if c >= rows)
                self.assertEqual(layout.capacity, capacity)
                factors = (1024, 1, 16 if dtype == torch.int64 else 8, 8)
                self.assertEqual(layout.lengths, tuple(rows * n for n in factors))
                self.assertEqual(layout.capacity_lengths, tuple(capacity * n for n in factors))
                signature = (layout.offsets, layout.capacity_lengths, layout.words)
                self.assertEqual(signatures.setdefault(capacity, signature), signature)
                end = 0
                for offset, reserved in zip(layout.offsets, layout.capacity_lengths):
                    self.assertEqual(offset * 4 % 128, 0)
                    self.assertGreaterEqual(offset, end)
                    end = offset + reserved
                self.assertGreaterEqual(layout.words, end)
                self.assertEqual(layout.nbytes, layout.words * 4)
                self.assertEqual(layout.nbytes % 128, 0)
                self.assertLessEqual(17 * layout.nbytes, 35 * 1024**2)
                if rows:
                    self.assertLess(capacity, 2 * rows)
                    self.assertLessEqual(layout.nbytes, 2 * rows * 4 * sum(factors))
            self.assertEqual(len(signatures), 10)
        self.assertEqual(self.helper.payload_layout(1, torch.int64).offsets, (0, 1024, 1056, 1088))
        self.assertEqual(self.helper.payload_layout(23, torch.int64).offsets, (0, 32768, 32800, 33312))
        self.assertEqual(self.helper.payload_layout(512, torch.int64).nbytes, 2148352)

    def test_packet_bits_and_all_unused_words_are_zero(self):
        # Repeated growth/shrink in capacity32, plus empty and boundary packets.
        for rows in (0, 1, 3, 17, 23, 31, 32, 17, 31, 23, 33, 257, 511, 512):
            for dtype in (torch.int64, torch.int32):
                with self.subTest(rows=rows, ids=dtype):
                    tensors = payload(rows, 5, dtype, strided=True)
                    packet = self.helper.pack_reference(*tensors)
                    layout = self.helper.payload_layout(rows, dtype)
                    used = torch.zeros(layout.words, dtype=torch.bool, device="cpu")
                    self.assertEqual(packet.dtype, torch.int32)
                    self.assertEqual(packet.numel(), layout.words)
                    for tensor, offset, length in zip(tensors, layout.offsets, layout.lengths):
                        self.assertEqual(raw_bytes(tensor).numel(), length * 4)
                        self.assertTrue(torch.equal(raw_bytes(packet[offset:offset + length]), raw_bytes(tensor)))
                        used[offset:offset + length] = True
                    self.assertFalse(bool(torch.any(packet[~used] != 0)))

    def test_unpack_exact_actual_rows_rank_order_and_contiguity(self):
        for rows in (0, 1, 3, 17, 23, 31, 32, 33, 511, 512):
            for dtype in (torch.int64, torch.int32):
                for strided in (False, True):
                    with self.subTest(rows=rows, ids=dtype, strided=strided):
                        peers = [payload(rows, rank, dtype, strided) for rank in range(16)]
                        gathered = torch.cat([self.helper.pack_reference(*p) for p in peers])
                        outputs = self.helper.unpack_reference(gathered, rows, dtype)
                        self.assertEqual(len(outputs), 4)
                        for field, actual in enumerate(outputs):
                            expected = torch.cat([p[field] for p in peers])
                            self.assert_bits_equal(actual, expected)
                            self.assertTrue(actual.is_contiguous())
                            self.assertEqual(actual.shape[0], 16 * rows)

    def test_nan_payloads_use_byte_equality(self):
        tensors = payload(23, 0, torch.int64, strided=True)
        packet = self.helper.pack_reference(*tensors)
        restored = self.helper.unpack_reference(packet, 23, torch.int64, world=1)
        self.assertTrue(bool(torch.isnan(tensors[1]).any()))
        self.assertFalse(torch.equal(restored[1], tensors[1]))  # NaN != NaN.
        self.assert_bits_equal(restored[1], tensors[1])

    def test_global_metadata_gate_includes_idle_peers_uniformly(self):
        for tokens in (0, 1, 7, 8, 9, 180, 4095, 4096, 4097, 8192):
            padded = ((tokens + 7) // 8 * 8,) * 2
            for original in ((tokens, 0), (0, tokens), (tokens, tokens)):
                metadata = (original, padded)
                decisions = [self.helper.target_metadata_eligible(16, metadata) for _ in range(16)]
                self.assertEqual(decisions, [0 < tokens <= 4096] * 16)
        for metadata in (None, ((0, 0), (8, 8)), ((1, 0), (7, 7)),
                         ((1, 0), (4104, 4104))):
            self.assertFalse(self.helper.target_metadata_eligible(16, metadata))
        for world in (1, 8, 32):
            self.assertFalse(self.helper.target_metadata_eligible(world, ((8, 0), (8, 8))))

    def test_malformed_shared_metadata_is_an_error(self):
        for metadata in (((1,), (8, 8)), ((1, 0), (8,)), ((True, 0), (8, 8)),
                         ((-1, 0), (8, 8)), ((1.0, 0), (8, 8)),
                         ((9, 0), (8, 8)), ((1, 0), (8, 16))):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.helper.target_metadata_eligible(16, metadata)

    def test_invalid_layout_payload_and_packet_rejected(self):
        for rows in (-1, 513, True, 1.0):
            with self.assertRaises(ValueError):
                self.helper.payload_layout(rows, torch.int64)
        with self.assertRaises(ValueError):
            self.helper.payload_layout(3, torch.int16)
        q, scale, ids, weights = payload(3, 0, torch.int64)
        with self.assertRaises(ValueError):
            self.helper.validate_payload(q, scale, ids, weights, expected_rows=4)
        for tensors in ((q[:, :-1], scale, ids, weights),
                        (q, scale.double(), ids, weights),
                        (q, scale, ids.to(torch.int16), weights),
                        (q, scale, ids, weights[:, :-1])):
            with self.assertRaises(ValueError):
                self.helper.pack_reference(*tensors)
        packet = self.helper.pack_reference(q, scale, ids, weights)
        with self.assertRaises(ValueError):
            self.helper.unpack_reference(packet[:-1], 3, torch.int64, world=1)


if __name__ == "__main__":
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
