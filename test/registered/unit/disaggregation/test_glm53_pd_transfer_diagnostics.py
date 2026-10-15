"""CPU-only checks of the actual poll/gate functions; no torch/NPU imports.

Run directly with python -B test/registered/unit/disaggregation/test_glm53_pd_transfer_diagnostics.py.
"""

import ast
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
FLAG = "SGLANG_GLM53_PD_TRANSFER_DIAGNOSTICS"


def load_functions():
    """Compile production bodies, stubbing only device/distributed boundaries."""
    path = SRT / "disaggregation/utils.py"
    tree = ast.parse(path.read_text())
    names = {
        "_apply_metadata_gate",
        "_log_pending_transfer_diagnostics",
        "poll_and_all_reduce",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    constants = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id.startswith("_PD_TRANSFER_DIAGNOSTIC_")
            for target in n.targets
        )
    ]
    base = ast.parse((SRT / "disaggregation/base/conn.py").read_text())
    poll_class = next(
        n for n in base.body if isinstance(n, ast.ClassDef) and n.name == "KVPoll"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            poll_class,
            *constants,
            *nodes,
        ],
        type_ignores=[],
    )
    scope = {
        "os": os,
        "json": json,
        "logger": Mock(),
        "time": NS(monotonic=Mock(return_value=0.0)),
        "_is_fake_transfer": lambda req: False,
        "_poll_with_failure_injection": lambda pollers: [p.poll() for p in pollers],
        "_all_reduce_polls": Mock(side_effect=lambda polls, group: list(polls)),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


class MarkerBuffer:
    def __init__(self, value=0):
        self.read = Mock(return_value=value)

    def __getitem__(self, index):
        return NS(item=self.read)


class TestPDTransferDiagnostics(unittest.TestCase):
    def setUp(self):
        self.scope = load_functions()
        self.poll = self.scope["KVPoll"]
        self.marker = MarkerBuffer()
        self.metadata = NS(bootstrap_room=self.marker)
        self.flag = patch.dict(os.environ, {FLAG: "1"})
        self.flag.start()
        self.addCleanup(self.flag.stop)

    def request(self, room=123, status=None, rank=0):
        status = self.poll.WaitingForInput if status is None else status
        acks = {0, 1, 2, 3} if status == self.poll.Success else {0, 1}
        manager = NS(
            prefill_response_tracker={room: acks},
            required_prefill_response_num_table={room: 4},
            request_status={room: status},
            attn_dp_rank=0,
            attn_tp_rank=rank,
            attn_tp_size=2,
        )
        receiver = NS(
            bootstrap_room=room,
            kv_mgr=manager,
            conclude_state=status if status == self.poll.Success else None,
            prefill_dp_rank=1,
            target_tp_ranks=list(range(4 * rank, 4 * rank + 4)),
            target_cp_ranks=[0],
            target_pp_ranks=[0],
            poll=Mock(return_value=status),
        )
        return NS(
            req=NS(rid=f"req-{room}", bootstrap_room=room),
            kv_receiver=receiver,
            metadata_buffer_index=7,
        )

    def run_poll(self, requests, now):
        self.scope["time"].monotonic.return_value = now
        return self.scope["poll_and_all_reduce"](
            [dr.kv_receiver for dr in requests], object(), requests, self.metadata
        )

    def records(self):
        return [
            json.loads(call.args[1])
            for call in self.scope["logger"].warning.call_args_list
        ]

    def test_waiting_ack_snapshot_delay_and_rate_limit_without_device_read(self):
        dr = self.request()
        for now in (0, 9.99):
            self.assertEqual(self.run_poll([dr], now), [self.poll.WaitingForInput])
        self.assertEqual(self.records(), [])
        self.run_poll([dr], 10)
        record = self.records()[0]
        self.assertEqual(record["ack_set"], [0, 1])
        self.assertEqual(record["expected_acks"], 4)
        self.assertEqual(record["raw_poll"], self.poll.WaitingForInput)
        self.assertEqual(record["prefill_dp_rank"], 1)
        self.assertEqual(record["target_tp_ranks"], [0, 1, 2, 3])
        self.assertFalse(record["metadata_sampled"])
        self.assertIsNone(record["metadata_room"])
        self.run_poll([dr], 69.99)
        self.assertEqual(len(self.records()), 1)
        self.run_poll([dr], 70)
        self.assertEqual(len(self.records()), 2)
        self.marker.read.assert_not_called()
        self.assertEqual(dr.kv_receiver.poll.call_count, 5)

    def test_success_with_zero_marker_reuses_exactly_one_native_read(self):
        dr = self.request(room=0, status=self.poll.Success)
        for now in (0, 10):
            self.assertEqual(self.run_poll([dr], now), [self.poll.Transferring])
        self.assertEqual(self.marker.read.call_count, 2)
        record = self.records()[0]
        self.assertEqual(record["room"], 0)
        self.assertEqual(record["request_room"], 0)
        self.assertEqual(record["metadata_index"], 7)
        self.assertEqual(record["ack_count"], 4)
        self.assertEqual(record["raw_poll"], self.poll.Success)
        self.assertEqual(record["local_poll"], self.poll.Transferring)
        self.assertEqual(record["reduced_poll"], self.poll.Transferring)
        self.assertTrue(record["metadata_sampled"])
        self.assertEqual(record["metadata_room"], 0)

    def test_locally_ready_peer_logs_after_tp_reduction(self):
        dr = self.request(status=self.poll.Success, rank=1)
        self.marker.read.return_value = 123
        self.scope["_all_reduce_polls"].side_effect = lambda polls, group: [
            self.poll.Transferring
        ]
        self.run_poll([dr], 0)
        self.run_poll([dr], 10)
        record = self.records()[0]
        self.assertEqual(record["local_poll"], self.poll.Success)
        self.assertEqual(record["reduced_poll"], self.poll.Transferring)
        self.assertEqual(record["metadata_room"], 123)
        self.assertEqual(record["attn_tp_rank"], 1)
        self.assertEqual(record["target_tp_ranks"], [4, 5, 6, 7])
        self.assertEqual(self.marker.read.call_count, 2)

    def test_disabled_preserves_gate_without_diagnostic_work(self):
        dr = self.request(status=self.poll.Success)
        with patch.dict(os.environ, {FLAG: "0"}):
            self.assertEqual(self.run_poll([dr], 100), [self.poll.Transferring])
        self.marker.read.assert_called_once()
        self.scope["time"].monotonic.assert_not_called()
        self.assertEqual(self.records(), [])
        self.assertEqual(self.scope["_PD_TRANSFER_DIAGNOSTIC_ROOMS"], {})

    def test_completed_and_removed_rooms_are_retired(self):
        dr = self.request()
        self.run_poll([dr], 0)
        dr.kv_receiver.poll.return_value = self.poll.Failed
        self.run_poll([dr], 10)
        self.assertEqual(self.scope["_PD_TRANSFER_DIAGNOSTIC_ROOMS"], {})
        self.assertEqual(self.records(), [])
        other = self.request(room=456)
        self.run_poll([other], 20)
        self.run_poll([], 30)
        self.assertEqual(self.scope["_PD_TRANSFER_DIAGNOSTIC_ROOMS"], {})

    def test_room_and_log_sample_are_bounded_to_256(self):
        requests = [self.request(room=i) for i in range(300)]
        self.run_poll(requests, 0)
        self.assertEqual(len(self.scope["_PD_TRANSFER_DIAGNOSTIC_ROOMS"]), 256)
        self.run_poll(requests, 10)
        self.assertEqual(len(self.records()), 256)
        self.assertEqual({r["room"] for r in self.records()}, set(range(256)))
        self.marker.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
