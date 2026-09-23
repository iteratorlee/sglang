import unittest
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
    envs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestProfilerStageCancel(unittest.TestCase):
    @staticmethod
    def manager(active):
        manager = object.__new__(SchedulerProfilerManager)
        manager.profile_in_progress = active
        manager.profile_by_stage = True
        manager.profiler_prefill_ct = 0
        manager.profiler_decode_ct = 0
        manager.profiler_target_prefill_ct = 4
        manager.profiler_target_decode_ct = 4
        manager.profiler_target_forward_ct = 100
        manager.profiler_start_forward_ct = 20
        manager._stop_profile = Mock(
            return_value=ProfileReqOutput(success=True, message="Succeeded")
        )
        return manager

    def test_cancel_disarms_idle_dp_domain(self):
        manager = self.manager(active=False)
        with patch.object(envs.SGLANG_PROFILE_V2, "get", return_value=False):
            result = manager._profile(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))
        self.assertTrue(result.success)
        manager._stop_profile.assert_not_called()
        self.assertFalse(manager.profile_by_stage)
        self.assertIsNone(manager.profiler_decode_ct)
        self.assertIsNone(manager.profiler_target_decode_ct)

    def test_cancel_stops_active_domain_and_disarms_it(self):
        manager = self.manager(active=True)
        with patch.object(envs.SGLANG_PROFILE_V2, "get", return_value=False):
            result = manager._profile(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))
        self.assertTrue(result.success)
        manager._stop_profile.assert_called_once_with()
        self.assertFalse(manager.profile_by_stage)
        self.assertIsNone(manager.profiler_target_forward_ct)


if __name__ == "__main__":
    unittest.main()
