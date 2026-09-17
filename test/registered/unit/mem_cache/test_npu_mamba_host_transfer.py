"""CPU-contract tests for the Ascend Mamba HiCache transfer path."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host import mamba as mamba_host
from sglang.srt.mem_cache.pool_host import npu_mamba_io
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuMambaTransferContract(unittest.TestCase):
    def test_layout_contract_preserves_component_dtype(self):
        temporal_device = torch.empty((3, 11, 2, 4, 5), dtype=torch.float32)
        temporal_host = torch.empty((13, 3, 1, 2, 4, 5), dtype=torch.float32)
        conv_device = torch.empty((3, 11, 7, 16), dtype=torch.bfloat16)
        conv_host = torch.empty((13, 3, 1, 7, 16), dtype=torch.bfloat16)

        npu_mamba_io._validate_component_layout(temporal_device, temporal_host)
        npu_mamba_io._validate_component_layout(conv_device, conv_host)

        with self.assertRaisesRegex(TypeError, "does not convert dtype"):
            npu_mamba_io._validate_component_layout(
                temporal_device, temporal_host.to(torch.bfloat16)
            )

    def test_layout_contract_rejects_wrong_page_or_payload(self):
        device = torch.empty((2, 8, 3, 5), dtype=torch.float32)
        wrong_page = torch.empty((9, 2, 2, 3, 5), dtype=torch.float32)
        wrong_payload = torch.empty((9, 2, 1, 3, 4), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "page size must be 1"):
            npu_mamba_io._validate_component_layout(device, wrong_page)
        with self.assertRaisesRegex(ValueError, "payload shapes differ"):
            npu_mamba_io._validate_component_layout(device, wrong_payload)

    def test_batched_dispatch_keeps_fp32_and_bf16_separate(self):
        temporal_device = torch.empty((2, 8, 3, 5), dtype=torch.float32)
        temporal_host = torch.empty((9, 2, 1, 3, 5), dtype=torch.float32)
        conv_device = torch.empty((2, 8, 4, 7), dtype=torch.bfloat16)
        conv_host = torch.empty((9, 2, 1, 4, 7), dtype=torch.bfloat16)
        device_indices = torch.tensor([7, 1, 5], dtype=torch.int64)
        host_indices = torch.tensor([2, 8, 3], dtype=torch.int64)
        calls = []

        class Direction:
            H2D = object()
            D2H = object()

        def native(*args):
            calls.append(args)

        with (
            mock.patch.object(
                npu_mamba_io, "_validate_runtime_component", return_value=None
            ),
            mock.patch.object(
                npu_mamba_io,
                "_native_transfer_objects",
                return_value=(Direction, native),
            ),
        ):
            npu_mamba_io.transfer_mamba_state_components(
                device_tensors=[temporal_device, conv_device],
                host_tensors=[temporal_host, conv_host],
                device_indices=device_indices,
                host_indices=host_indices,
                direction="d2h",
            )

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], temporal_device)
        self.assertIs(calls[0][1], temporal_host)
        self.assertIs(calls[1][0], conv_device)
        self.assertIs(calls[1][1], conv_host)
        self.assertIs(calls[0][-1], Direction.D2H)
        self.assertIs(calls[1][-1], Direction.D2H)

    def test_empty_temporal_component_is_skipped(self):
        temporal_device = torch.empty((2, 8, 0), dtype=torch.float32)
        temporal_host = torch.empty((9, 2, 1, 0), dtype=torch.float32)
        conv_device = torch.empty((2, 8, 4, 7), dtype=torch.bfloat16)
        conv_host = torch.empty((9, 2, 1, 4, 7), dtype=torch.bfloat16)
        native = mock.Mock()

        class Direction:
            H2D = object()
            D2H = object()

        with (
            mock.patch.object(
                npu_mamba_io, "_validate_runtime_component", return_value=None
            ),
            mock.patch.object(
                npu_mamba_io,
                "_native_transfer_objects",
                return_value=(Direction, native),
            ),
        ):
            npu_mamba_io.transfer_mamba_state_components(
                device_tensors=[temporal_device, conv_device],
                host_tensors=[temporal_host, conv_host],
                device_indices=torch.tensor([1, 4]),
                host_indices=torch.tensor([3, 6]),
                direction="h2d",
            )

        native.assert_called_once()
        self.assertIs(native.call_args.args[0], conv_device)
        self.assertIs(native.call_args.args[-1], Direction.H2D)

    def test_index_contract_fails_before_native_submission(self):
        native = mock.Mock()
        with mock.patch.object(
            npu_mamba_io,
            "_native_transfer_objects",
            return_value=(mock.Mock(), native),
        ):
            with self.assertRaisesRegex(ValueError, "index counts differ"):
                npu_mamba_io.transfer_mamba_state_components(
                    device_tensors=[],
                    host_tensors=[],
                    device_indices=torch.tensor([1, 2]),
                    host_indices=torch.tensor([3]),
                    direction="h2d",
                )
        native.assert_not_called()


class TestMambaPoolHostAscendDispatch(unittest.TestCase):
    @staticmethod
    def _pool_and_device():
        host = mamba_host.MambaPoolHost.__new__(mamba_host.MambaPoolHost)
        host.temporal_buffer = torch.empty((9, 2, 1, 3), dtype=torch.float32)
        host.conv_buffer = [
            torch.empty((9, 2, 1, 4, 7), dtype=torch.bfloat16)
        ]
        device = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.empty((2, 8, 3), dtype=torch.float32),
                conv=[torch.empty((2, 8, 4, 7), dtype=torch.bfloat16)],
            )
        )
        return host, device

    def test_h2d_submits_all_layers_once(self):
        host, device = self._pool_and_device()
        transfer = mock.Mock()
        host_indices = torch.tensor([2, 6])
        device_indices = torch.tensor([7, 1])

        with (
            mock.patch.object(mamba_host, "_is_npu", True),
            mock.patch.object(
                mamba_host,
                "transfer_mamba_state_components",
                transfer,
                create=True,
            ),
        ):
            host.load_to_device_per_layer(
                device, host_indices, device_indices, 0, "kernel_ascend"
            )
            host.load_to_device_per_layer(
                device, host_indices, device_indices, 1, "kernel_ascend"
            )

        transfer.assert_called_once()
        self.assertEqual(transfer.call_args.kwargs["direction"], "h2d")
        self.assertIs(
            transfer.call_args.kwargs["device_tensors"][0],
            device.mamba_cache.temporal,
        )

    def test_d2h_submits_all_components(self):
        host, device = self._pool_and_device()
        transfer = mock.Mock()

        with (
            mock.patch.object(mamba_host, "_is_npu", True),
            mock.patch.object(
                mamba_host,
                "transfer_mamba_state_components",
                transfer,
                create=True,
            ),
        ):
            host.backup_from_device_all_layer(
                device,
                torch.tensor([2, 6]),
                torch.tensor([7, 1]),
                "kernel_ascend",
            )

        transfer.assert_called_once()
        self.assertEqual(transfer.call_args.kwargs["direction"], "d2h")
        self.assertEqual(len(transfer.call_args.kwargs["device_tensors"]), 2)

    def test_npu_destroy_drops_named_pinned_buffer_owners(self):
        host, _ = self._pool_and_device()
        host.pin_memory = True
        host.kv_buffer = [host.temporal_buffer, *host.conv_buffer]
        host.temporal_staging_buffer = None
        host.conv_staging_buffers = [None]
        host.temporal_device_ptrs = None
        host.conv_device_ptrs = [None]

        with mock.patch.object(mamba_host, "_is_npu", True):
            host.destroy()

        self.assertIsNone(host.kv_buffer)
        self.assertIsNone(host.temporal_buffer)
        self.assertEqual(host.conv_buffer, [])


if __name__ == "__main__":
    unittest.main()
