import unittest
from types import SimpleNamespace
import torch
from go2_hindstand_lab.reference_rewards import ReferenceRewards

class EventIntegralTests(unittest.TestCase):
    def test_landing_integral_is_control_timestep_invariant(self):
        for dt in (.01,.02,.04):
            s=ReferenceRewards();s.dt=dt;s.gait=SimpleNamespace(air_score=torch.tensor([-.5,0.,.5,1.]))
            s.cfg=SimpleNamespace(rewards=SimpleNamespace(air_time_event_integral=True))
            torch.testing.assert_close(s._reward_feet_air_time()*dt,s.gait.air_score)
    def test_disabled_mode_preserves_historical_event_rate(self):
        s=ReferenceRewards();s.dt=.02;s.gait=SimpleNamespace(air_score=torch.tensor([-.5,0.,.5]))
        s.cfg=SimpleNamespace(rewards=SimpleNamespace(air_time_event_integral=False))
        torch.testing.assert_close(s._reward_feet_air_time(),s.gait.air_score)

if __name__=='__main__':unittest.main()
