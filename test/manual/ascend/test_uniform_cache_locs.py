"""CPU-oracle and graph replay checks for NPU speculative KV locations.

Run in the Ascend image with its CANN environment sourced.
"""

import unittest

import torch
import torch_npu

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_uniform_func,
)


class TestUniformCacheLocs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.length = 132160
        cls.pool_cpu = torch.arange(17 * cls.length, dtype=torch.int32).reshape(
            17, cls.length
        )
        cls.pool = cls.pool_cpu.to("npu")

    def expected(self, requests, starts, width):
        return torch.cat(
            [self.pool_cpu[r, s : s + width] for r, s in zip(requests, starts)]
        )

    def test_exact_gather_at_long_context_and_row_boundaries(self):
        for width in (1, 3, 4, 8, 16, 17, 65):
            for bs in (1, 2, 3, 7, 8, 9, 10, 11, 12, 16):
                with self.subTest(width=width, batch=bs):
                    # For B16 the final physical pool row also gets the
                    # row-end start, exercising the last valid pool element.
                    requests = torch.roll(
                        torch.arange(bs, 0, -1, dtype=torch.int64), 3
                    )
                    starts = torch.tensor(
                        [0, 65536, 131072, self.length - width] * 4,
                        dtype=torch.int64,
                    )[:bs]
                    result = assign_extend_cache_locs_uniform_func(
                        requests.to("npu"), self.pool, starts.to("npu"), bs, width, "npu"
                    )
                    self.assertEqual(result.dtype, torch.int32)
                    self.assertEqual(tuple(result.shape), (bs * width,))
                    torch.testing.assert_close(
                        result.cpu(), self.expected(requests, starts, width), rtol=0, atol=0
                    )

    def test_graph_replay_uses_current_request_and_position(self):
        bs, width = 10, 4
        requests = torch.arange(1, bs + 1, dtype=torch.int64, device="npu")
        starts = torch.full((bs,), 65536, dtype=torch.int64, device="npu")
        for _ in range(3):
            result = assign_extend_cache_locs_uniform_func(
                requests, self.pool, starts, bs, width, "npu"
            )
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = assign_extend_cache_locs_uniform_func(
                requests, self.pool, starts, bs, width, "npu"
            )
        output_ptr = result.data_ptr()
        for iteration in range(64):
            req_cpu = torch.roll(torch.arange(1, bs + 1), iteration % bs)
            start_cpu = torch.tensor(
                [65536 + iteration, 131072 + iteration, self.length - width, 0] * 3,
                dtype=torch.int64,
            )[:bs]
            requests.copy_(req_cpu)
            starts.copy_(start_cpu)
            graph.replay()
            self.assertEqual(result.data_ptr(), output_ptr)
            torch.testing.assert_close(
                result.cpu(), self.expected(req_cpu, start_cpu, width), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
