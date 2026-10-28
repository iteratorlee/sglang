import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("payload",ROOT/"scripts/package_image_payload.py")
payload=importlib.util.module_from_spec(spec)
spec.loader.exec_module(payload)


class CommittedPayload(unittest.TestCase):
    def test_allowlist_export_records_commit_and_sha_without_any_worktree_reads(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            output=Path(directory)/"payload"
            calls=[]
            def blob(repo,commit,path):
                calls.append(path)
                return b"adapter" if path.endswith("pd_monitoring.py") else b"source"
            with patch.object(payload.subprocess,"check_output",return_value="a"*40+"\n"), patch.object(payload,"read_blob",side_effect=blob):
                result=payload.export(Path("/nonexistent-worktree"),"HEAD",output)
            self.assertEqual(result["adapter_commit"],"a"*40)
            self.assertEqual(len(result["adapter_sha256"]),64)
            self.assertEqual(set(result["files"]),set(payload.COPY_FILES))
            self.assertEqual(len(calls),len(payload.COPY_FILES)+1)
            for name in result["files"]:
                self.assertNotIn(".git",name);self.assertNotIn("tests",name);self.assertNotIn("artifacts",name)
            self.assertTrue((output/"monitoring/package-version.json").is_file())

    def test_adapter_drift_fails_before_any_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            output=Path(directory)/"payload"
            with patch.object(payload.subprocess,"check_output",return_value="a"*40+"\n"), patch.object(payload,"read_blob",side_effect=lambda repo,commit,path:path.encode()):
                with self.assertRaises(ValueError):payload.export(Path("/repo"),"HEAD",output)
            self.assertFalse(output.exists())

    def test_runtime_bindings_and_private_machine_config_are_not_image_payload(self):
        self.assertEqual(set(payload.COPY_FILES), {
            "monitoring/sglang_pd_monitoring.py", "monitoring/exporter/exporter.py",
            "source-hooks/runtime-prefill-rank.patch", "source-hooks/runtime-decode-rank.patch"})
        exporter=(ROOT/"exporter/exporter.py").read_text()
        self.assertNotIn("10.0.0.",exporter)
        self.assertNotIn("910b-16",exporter)
        self.assertIn('required=True, help="Explicit deployment topology',exporter)
