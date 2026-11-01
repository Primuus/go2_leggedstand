import unittest
from types import SimpleNamespace
import torch
from deploy_mujoco.config_loader import load_config
from go2_hindstand_lab.swing_reference import swing_reference,swing_reference_score
from go2_hindstand_lab.gait_events import GaitEvents

class SwingReferenceTests(unittest.TestCase):
    def cfg(self,mode=True):
        c=load_config().rewards
        d=SimpleNamespace(**{k:getattr(c,k) for k in dir(c) if not k.startswith('_')})
        d.gait_use_swing_reference=mode
        return d
    def test_contact_timed_endpoints_and_mid_swing(self):
        t=torch.tensor([[0.,.12,.24]])
        h,x=swing_reference(t,torch.full_like(t,-.03),torch.full_like(t,.06),.06,.24)
        torch.testing.assert_close(h,torch.tensor([[0.,.06,0.]]),atol=1e-7,rtol=0)
        torch.testing.assert_close(x,torch.tensor([[-.03,.015,.06]]),atol=1e-7,rtol=0)
    def test_matching_height_cannot_compensate_for_missing_placement(self):
        c=self.cfg();t=torch.tensor([[.12]]);a=torch.tensor([[-.03]]);target=torch.tensor([[.06]]);linear=torch.tensor([[True]])
        def score(h,x):return float(swing_reference_score(t,torch.tensor([[h]]),torch.tensor([[x]]),a,target,linear,c))
        self.assertAlmostEqual(score(.06,.015),1.,places=6)
        self.assertLess(score(0.,.015),.001)
        self.assertLess(score(.06,-.075),.001)
        self.assertLess(score(0.,-.075),.001)
    def test_turning_does_not_require_forward_placement(self):
        score=swing_reference_score(torch.tensor([[.12]]),torch.tensor([[.06]]),torch.tensor([[-1.]]),torch.tensor([[0.]]),torch.tensor([[0.]]),torch.tensor([[False]]),self.cfg())
        torch.testing.assert_close(score,torch.ones(1,1))
    def test_shaping_does_not_change_observable_state_or_landing_history(self):
        a,b=GaitEvents(1,'cpu',.02,self.cfg(True)),GaitEvents(1,'cpu',.02,self.cfg(False))
        seq=[([75.,75.],[0.,0.],[0.,0.])]+[([150.,0.],[0.,.04],[0.,.03])]*10+[([75.,75.],[0.,0.],[0.,.06])]*3
        scores=[]
        for forces,heights,forward in seq:
            for s in (a,b):s.update(torch.tensor([forces]),torch.tensor([heights]),torch.tensor([True]),torch.tensor([150.]),torch.tensor([forward]),torch.tensor([.2]))
            torch.testing.assert_close(a.observation(torch.tensor([True])),b.observation(torch.tensor([True])))
            torch.testing.assert_close(a.completed,b.completed)
            scores.append(float(a.clearance_score-b.clearance_score))
        self.assertTrue(any(abs(x)>1e-5 for x in scores))
        self.assertTrue(a.last_touch[0,1])

if __name__=='__main__':unittest.main()
