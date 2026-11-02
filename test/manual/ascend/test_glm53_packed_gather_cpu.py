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
from types import ModuleType
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

torch.set_num_threads(1)

HELPER = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/moe/glm53_packed_gather.py"
)
HELPER = Path(os.environ.get("SGLANG_GLM53_PACKED_HELPER_SOURCE", HELPER))


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
        spec = importlib.util.spec_from_file_location(name, HELPER, submodule_search_locations=[])
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
                self.assertLessEqual(33 * layout.nbytes, 68 * 1024**2)
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

    def test_ep32_packet_parity_including_last_rank_and_expert_boundaries(self):
        for rows in (1, 3, 17, 31, 32, 33, 257, 511, 512):
            for dtype in (torch.int64, torch.int32):
                with self.subTest(rows=rows, ids=dtype):
                    peers = [payload(rows, rank, dtype, strided=True) for rank in range(32)]
                    # Packing is independent of expert ownership; retain IDs at
                    # both sides of the EP32 nine-expert boundaries verbatim.
                    peers[-1][2][0] = torch.tensor(
                        [0, 8, 9, 17, 18, 278, 279, 287], dtype=dtype
                    )
                    gathered = torch.cat([self.helper.pack_reference(*p) for p in peers])
                    outputs = self.helper.unpack_reference(gathered, rows, dtype, world=32)
                    for field, actual in enumerate(outputs):
                        self.assert_bits_equal(actual, torch.cat([p[field] for p in peers]))
                        self.assertTrue(actual.is_contiguous())
                        self.assertEqual(actual.shape[0], 32 * rows)

    def test_production_wrapper_one_collective_and_world_sized_outputs(self):
        # Run the actual allocation/collective wrapper on CPU. Only NPU format
        # validation, transport and Triton launches are replaced by CPU oracles.
        helper = self.helper
        validate = helper.validate_payload
        kernels = ModuleType(helper.__name__ + ".glm53_packed_gather_triton")

        def cpu_validate(*args, **kwargs):
            kwargs.pop("npu", None)
            return validate(*args, **kwargs)

        def pack(q, scale, ids, weights, send, layout):
            send.copy_(helper.pack_reference(q, scale, ids, weights))

        def unpack(gathered, outputs, layout, world):
            for actual, expected in zip(outputs, helper.unpack_reference(
                gathered, layout.rows, outputs[2].dtype, world=world
            )):
                actual.copy_(expected)

        kernels.pack, kernels.unpack = mock.Mock(side_effect=pack), mock.Mock(side_effect=unpack)
        with mock.patch.dict(sys.modules, {kernels.__name__: kernels}), \
             mock.patch.object(helper, "validate_payload", side_effect=cpu_validate):
            for world in (16, 32):
                for dtype in (torch.int64, torch.int32):
                    peers = [payload(17, rank, dtype, strided=True) for rank in range(world)]
                    group = object()

                    def gather(out, send, *, group):
                        self.assert_bits_equal(send, helper.pack_reference(*peers[0]))
                        out.copy_(torch.cat([helper.pack_reference(*p) for p in peers]))

                    before = helper.get_packed_gather_stats()["packed_gather_calls"]
                    with mock.patch.object(helper.dist, "get_world_size", return_value=world), \
                         mock.patch.object(helper.dist, "all_gather_into_tensor", side_effect=gather) as collective:
                        outputs = helper.packed_all_gather(group, *peers[0], expected_rows=17)
                    self.assertEqual(collective.call_count, 1)
                    self.assertIs(collective.call_args.kwargs["group"], group)
                    self.assertEqual(kernels.unpack.call_args.args[-1], world)
                    self.assertEqual(helper.get_packed_gather_stats()["packed_gather_calls"], before + 1)
                    for field, actual in enumerate(outputs):
                        self.assert_bits_equal(actual, torch.cat([p[field] for p in peers]))

            for world, rows in ((1, 1), (8, 1), (24, 1), (64, 1), (32, 0)):
                with mock.patch.object(helper.dist, "get_world_size", return_value=world), \
                     mock.patch.object(helper.dist, "all_gather_into_tensor") as collective:
                    with self.assertRaises(ValueError):
                        helper.packed_all_gather(object(), *payload(rows, 0, torch.int64), expected_rows=rows)
                    collective.assert_not_called()

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
        for world in (1, 8, 24, 64):
            self.assertFalse(self.helper.target_metadata_eligible(world, ((8, 0), (8, 8))))

    def test_ep32_all_idle_masks_and_bounded_global_metadata(self):
        for tokens in (0, 1, 7, 8, 9, 128, 256, 4095, 4096, 4097, 16384):
            padded = ((tokens + 7) // 8 * 8,) * 4
            for mask in range(16):
                original = tuple(tokens if mask & (1 << dp) else 0 for dp in range(4))
                decisions = [self.helper.target_metadata_eligible(32, (original, padded)) for _ in range(32)]
                self.assertEqual(decisions, [bool(mask) and 0 < tokens <= 4096] * 32)
        self.assertTrue(self.helper.target_metadata_eligible(32, ((1, 128, 0, 255), (256,) * 4)))
        for metadata in (None, ((0,) * 4, (8,) * 4), ((1, 0, 0, 0), (7,) * 4),
                         ((1, 0, 0, 0), (4104,) * 4)):
            self.assertFalse(self.helper.target_metadata_eligible(32, metadata))
        for metadata in (((1, 0), (8, 8)), ((1, 0, 0, 0), (8, 8)),
                         ((True, 0, 0, 0), (8,) * 4), ((-1, 0, 0, 0), (8,) * 4),
                         ((1.0, 0, 0, 0), (8,) * 4), ((9, 0, 0, 0), (8,) * 4),
                         ((1, 0, 0, 0), (8, 8, 16, 8)), ((1, 0, 0, 0), (8, 8, 8, 16))):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.helper.target_metadata_eligible(32, metadata)

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
