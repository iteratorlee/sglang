"""CPU checks for the Gateway's selected decode-rank request metadata."""

from array import array
import unittest

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session


class TestDecodeRankContract(unittest.TestCase):
    def test_single_and_batch_rank_selection(self):
        single = GenerateReqInput(input_ids=[1, 2], disagg_decode_dp_rank=3)
        single.normalize_batch_and_arguments()
        self.assertEqual(single.disagg_decode_dp_rank, 3)
        batch = GenerateReqInput(
            input_ids=[[1], [2]],
            sampling_params=[{}, {}],
            disagg_decode_dp_rank=[3, 5],
        )
        batch.normalize_batch_and_arguments()
        self.assertEqual([batch[i].disagg_decode_dp_rank for i in range(2)], [3, 5])

    def test_parallel_sampling_preserves_input_ranks(self):
        for ids, params, ranks, expected in (
            ([1, 2], {"n": 3}, 3, [3, 3, 3]),
            ([[1], [2]], [{"n": 2}, {"n": 2}], [3, 5], [3, 5, 3, 5]),
        ):
            with self.subTest(ranks=ranks):
                req = GenerateReqInput(
                    input_ids=ids, sampling_params=params,
                    disagg_decode_dp_rank=ranks,
                )
                req.normalize_batch_and_arguments()
                self.assertEqual(
                    [req[i].disagg_decode_dp_rank for i in range(len(expected))],
                    expected,
                )

    def test_invalid_rank_and_batch_length_are_rejected(self):
        for rank in (-1, True, "3", [3]):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                GenerateReqInput(
                    input_ids=[1], disagg_decode_dp_rank=rank,
                ).normalize_batch_and_arguments()
        with self.assertRaises(ValueError):
            GenerateReqInput(
                input_ids=[[1], [2]], disagg_decode_dp_rank=[3],
            ).normalize_batch_and_arguments()

    @staticmethod
    def tokenized(session_params=None):
        return TokenizedGenerateReqInput(
            rid="decode-rank-contract", input_text="",
            input_ids=array("q", [101, 102]), input_embeds=None,
            mm_inputs=None, token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=16),
            return_logprob=False, logprob_start_len=0, top_logprobs_num=0,
            token_ids_logprob=None, stream=False, session_params=session_params,
            disagg_decode_dp_rank=7,
        )

    def test_ipc_preserves_rank_and_accepts_legacy_array(self):
        req = self.tokenized()
        encoded = msgpack_encode(req)
        self.assertEqual(msgpack_decode(encoded).disagg_decode_dp_rank, 7)
        raw = msgspec.msgpack.decode(encoded)
        self.assertEqual(raw[-1], 7)
        legacy = msgpack_decode(msgspec.msgpack.encode(raw[:-1]))
        self.assertIsNone(legacy.disagg_decode_dp_rank)
        self.assertEqual(legacy.rid, req.rid)
        self.assertEqual(list(legacy.input_ids), [101, 102])

    def test_session_request_preserves_selected_rank(self):
        session = Session(capacity_of_str_len=0, session_id="rank-test", streaming=False)
        tokenized = self.tokenized(SessionParams(id=session.session_id))
        req = session.create_req(
            tokenized, tokenizer=None, vocab_size=4096, eos_token_ids={2},
            disagg_mode=DisaggregationMode.PREFILL,
        )
        self.assertEqual(req.disagg_decode_dp_rank, 7)
        self.assertIs(req.session, session)
        self.assertIs(session.req_nodes[req.rid].req, req)


if __name__ == "__main__":
    unittest.main()
