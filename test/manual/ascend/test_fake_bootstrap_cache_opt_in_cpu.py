"""Exercise the actual Req constructor's fake-bootstrap cache opt-in on CPU."""

from array import array
import copy
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

import test_fake_bootstrap_chunk_progress_cpu as lifecycle

FLAG = "SGLANG_FAKE_BOOTSTRAP_ENABLE_RADIX_CACHE"
FAKE_HOST = "2.2.2.2"
N = dict(
    lifecycle.N,
    os=os,
    copy=copy,
    FAKE_BOOTSTRAP_HOST=FAKE_HOST,
    DecLockRefParams=lambda: object(),
    get_return_hidden_states_mode=lambda value: None,
    SchedulerReqTimeStats=Mock(),
)
lifecycle.load(lifecycle.BATCH, "ReqLogprob", N)


class Request:
    # Diffusion setup and wall-clock accounting are unrelated constructor
    # dependencies. The complete production Req.__init__ executes unchanged.
    def init_diffusion_llm(self, config):
        self.dllm_config = config


Request.__init__ = lifecycle.load(lifecycle.BATCH, "__init__", N, "Req")


def request(host, value=None):
    with patch.dict(os.environ):
        if value is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = value
        return Request(
            rid="cpu-cache-policy",
            origin_input_text=None,
            origin_input_ids=array("q", [1, 2, 3]),
            sampling_params=NS(custom_params=None, max_new_tokens=1),
            bootstrap_host=host,
        )


class FakeBootstrapCacheOptInTests(unittest.TestCase):
    def test_fake_bootstrap_default_keeps_skip_policy(self):
        req = request(FAKE_HOST)
        self.assertTrue(req.skip_radix_cache_insert)
        self.assertEqual(req.bootstrap_host, FAKE_HOST)
        self.assertIsNone(req.last_node)
        self.assertEqual(req.kv.cache_protected_len, 0)
        self.assertEqual(req.prefix_indices.device.type, "cpu")
        self.assertEqual(req.prefix_indices.dtype, torch.int64)

    def test_only_literal_one_enables_normal_cache_delegation(self):
        req = request(FAKE_HOST, "1")
        self.assertFalse(req.skip_radix_cache_insert)
        cache = NS(cache_unfinished_req=Mock())
        lifecycle.maybe_cache(req, cache, chunked=True)
        cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)

    def test_other_spellings_keep_fake_bootstrap_skipped(self):
        for value in ("", "0", "true", "True", "yes", "01", "1 ", " 1", "1\n"):
            with self.subTest(value=value):
                self.assertTrue(request(FAKE_HOST, value).skip_radix_cache_insert)

    def test_ordinary_requests_always_use_normal_cache(self):
        for host in (None, "127.0.0.1", "192.0.2.1"):
            for value in (None, "0", "1", "true"):
                with self.subTest(host=host, value=value):
                    req = request(host, value)
                    self.assertFalse(req.skip_radix_cache_insert)
                    cache = NS(cache_unfinished_req=Mock())
                    lifecycle.maybe_cache(req, cache)
                    cache.cache_unfinished_req.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
