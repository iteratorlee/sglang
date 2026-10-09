"""Test actual callback: skip-warmup listener race, auth and bounded failure."""
import ast
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests


class TestFreezeAfterWarmup(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / 'python/sglang/srt/entrypoints/http_server.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_freeze_gc_after_server_warmup')
        self.post = Mock()
        self.time = Mock()
        self.time.monotonic.return_value = 0
        self.logger = Mock()
        self.args = types.SimpleNamespace(url=lambda: 'http://127.0.0.1:31116')
        self.serving = types.SimpleNamespace(admin_api_key='admin-key', api_key='user-key')
        scope = {'ServerArgs': object, 'requests': types.SimpleNamespace(
            post=self.post, exceptions=requests.exceptions), 'time': self.time,
            'logger': self.logger, 'get_serving': lambda: self.serving,
            'ssl_verify_of': lambda args: False}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
        self.run_callback = lambda: scope['_freeze_gc_after_server_warmup'](self.args)

    def test_success_preserves_admin_auth_and_ssl(self):
        self.run_callback()
        self.post.assert_called_once_with('http://127.0.0.1:31116/freeze_gc',
            headers={'Authorization': 'Bearer admin-key'}, timeout=10, verify=False)
        self.time.sleep.assert_not_called()
        self.logger.warning.assert_not_called()

    def test_listener_race_recovers_before_freeze(self):
        self.post.side_effect = [requests.ConnectionError('refused'),
                                requests.ConnectionError('refused'), Mock()]
        self.run_callback()
        self.assertEqual(self.post.call_count, 3)
        self.assertEqual(self.time.sleep.call_count, 2)
        self.logger.warning.assert_not_called()

    def test_deadline_failure_is_bounded(self):
        self.time.monotonic.side_effect = [0, 1, 11]
        self.post.side_effect = requests.ConnectionError('refused')
        self.run_callback()
        self.assertEqual(self.post.call_count, 2)
        self.time.sleep.assert_called_once_with(0.25)
        self.logger.warning.assert_called_once()

    def test_http_rejection_is_not_retried(self):
        self.post.return_value.raise_for_status.side_effect = requests.HTTPError('403')
        self.run_callback()
        self.post.assert_called_once()
        self.time.sleep.assert_not_called()
        self.logger.warning.assert_called_once()


if __name__ == '__main__':
    unittest.main()
