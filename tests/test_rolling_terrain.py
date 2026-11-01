"""Geometry, height semantics and advancement checks for continuous rolling terrain."""
import unittest
from pathlib import Path
import numpy as np
import torch
from go2_hindstand_lab.terrain_profile import profile, GroundProfiles, SLOPE_DEGREES, SPACING, TerrainSettings, initial_terrain_assignment
from go2_hindstand_lab.terrain_curriculum import RollingCurriculum
from go2_hindstand_lab.reference_rewards import ReferenceRewards
from deploy_mujoco.config_loader import load_config


class RollingTests(unittest.TestCase):
    def test_all_l1_has_no_flat_lanes_and_cannot_demote_to_l0(self):
        settings = TerrainSettings(mode="rolling", min_level=1, initial_level=1, max_level=1, keep_flat=False)
        levels, columns = initial_terrain_assignment(torch.arange(64)%10, settings)
        self.assertTrue((levels == 1).all());self.assertTrue((columns >= 5).all())
        self.assertEqual(len(columns.unique()),5)
        c = RollingCurriculum(64,"cpu",max_level=1,min_level=1)
        ids = torch.arange(64)
        for _ in range(40):
            c.moving_seconds[:]=2.;c.distance[:]=.4;c.progress[:]=0.
            up,down=c.finish(ids,levels,columns,torch.ones(64,dtype=torch.bool))
            self.assertFalse(up.any());self.assertFalse(down.any())
            c.reset_episode(ids)
        mixed = TerrainSettings(mode="rolling",initial_level=1)
        l,k = initial_terrain_assignment(torch.arange(10),mixed)
        torch.testing.assert_close(l,torch.tensor([0,0,0,0,0,1,1,1,1,1]))
        with self.assertRaises(ValueError):TerrainSettings(min_level=2,initial_level=1)

    def test_profiles_are_continuous_bounded_slopes_with_flat_spawn_and_both_signs(self):
        last = 0.
        for level, angle in enumerate(SLOPE_DEGREES):
            x, z = profile(level, 5, seed=3)
            self.assertLessEqual(np.max(np.abs(np.diff(z)/SPACING)), np.tan(np.deg2rad(angle))+1e-10)
            np.testing.assert_array_equal(z[np.abs(x) <= 1.], 0.)
            if level:
                self.assertLess(z.min(), 0)
                self.assertGreater(z.max(), 0)
                self.assertGreater(z.max()-z.min(), last)
            last = z.max()-z.min()
        np.testing.assert_array_equal(profile(4, 0)[1], 0.)
        self.assertFalse(np.array_equal(profile(2, 5, 1)[1], profile(2, 5, 2)[1]))

    def test_torch_queries_match_piecewise_linear_surface_and_footprint(self):
        g = GroundProfiles(seed=3)
        x = torch.linspace(-11, 11, 101).reshape(1, -1)
        actual = g.height(x, torch.tensor([3]), torch.tensor([5])).numpy()[0]
        grid, z = profile(3, 5, 3)
        
        # Float32 grid coordinates at steep slopes: require sub-micrometre agreement.
        np.testing.assert_allclose(actual, np.interp(x.numpy()[0], grid, z), atol=1e-6)
        self.assertTrue((g.sole_ground(x, torch.tensor([3]), torch.tensor([5])) >= torch.tensor(actual)).all())

    def test_height_reward_is_relative_and_independent_between_environments(self):
        s = ReferenceRewards();s.cfg=load_config()
        s.root_states = torch.zeros(2,13);s.root_states[:,2]=torch.tensor([.52,1.52])
        s.measured_heights = torch.tensor([[0.],[1.]])
        torch.testing.assert_close(s._reward_base_height(), torch.ones(2))
        s.root_states[1,2]=1.1;s._reward_base_height()
        s.commands=torch.zeros(2,4);s.base_lin_vel=torch.zeros(2,3)
        torch.testing.assert_close(s._reward_tracking_lin_vel(), torch.tensor([1.,0.]))

    def test_standing_does_not_promote_and_lateral_motion_cannot_pass(self):
        c=RollingCurriculum(2,'cpu');ids=torch.arange(2);levels=torch.zeros(2,dtype=torch.long);cols=torch.tensor([5,6])
        for _ in range(45):
            c.advance(torch.tensor([[0.,0.,0.],[0.,1.,0.]]), torch.zeros(2), .02)
            up,down=c.finish(ids,levels,cols,torch.zeros(2,dtype=torch.bool));c.reset_episode(ids)
            self.assertFalse(up.any());self.assertFalse(down.any())
        self.assertEqual(c.count.sum(),0)
        c.distance[:]=1.;c.moving_seconds[:]=5;c.progress[:]=1.;c.max_lateral[1]=1.
        c.finish(ids,levels,cols,torch.zeros(2,dtype=torch.bool))
        self.assertTrue(c.success[0,0]);self.assertFalse(c.success[1,0])

    def test_promotion_demotion_cap_flat_reservation_and_restore(self):
        c=RollingCurriculum(3,'cpu');ids=torch.arange(3);levels=torch.tensor([0,1,4]);cols=torch.tensor([0,5,6])
        for i in range(40):
            c.distance[:]=1;c.moving_seconds[:]=5;c.progress[:]=1
            up,down=c.finish(ids,levels,cols,torch.zeros(3,dtype=torch.bool));c.reset_episode(ids)
            self.assertEqual(bool(up[1]),i==39)
        self.assertFalse(up[0]);self.assertFalse(up[2]);self.assertFalse(down.any())
        for i in range(2):
            c.distance[:]=1;c.moving_seconds[:]=5;c.progress[:]=.1
            up,down=c.finish(ids,levels,cols,torch.ones(3,dtype=torch.bool));c.reset_episode(ids)
        self.assertTrue(down[1]);self.assertFalse(down[0])
        d=RollingCurriculum(3,'cpu');d.load_state_dict(c.state_dict())
        for k,v in c.state_dict().items():torch.testing.assert_close(d.state_dict()[k],v)

    def test_short_forward_falls_count_but_standing_falls_do_not(self):
        c=RollingCurriculum(2,'cpu'); ids=torch.arange(2); levels=torch.ones(2,dtype=torch.long); cols=torch.tensor([5,6])
        for _ in range(2):
            c.moving_seconds[0]=.04;c.distance[0]=.008
            up,down=c.finish(ids,levels,cols,torch.ones(2,dtype=torch.bool));c.reset_episode(ids)
        self.assertTrue(down[0]);self.assertFalse(down[1]);self.assertEqual(c.count[1],0)

    def test_mujoco_heightfield_contact_is_measured_as_ground_contact(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('MuJoCo is evaluated locally')
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy_mujoco'))
        from sim2sim_go2_handstand import Runner
        r=Runner(terrain='rolling',terrain_level=2)
        r.data.qpos[0] += 1.5
        mujoco.mj_forward(r.model,r.data)
        r.data.qpos[2] += r.ground_heights().mean()
        mujoco.mj_forward(r.model,r.data)
        for _ in range(8):r.step()
        heights,forces=r.feet_state()
        self.assertGreater(forces.sum(),1.)
        self.assertLess(np.max(np.abs(heights)),.02)
        self.assertEqual(r.model.geom_type[r.model.geom('floor').id],mujoco.mjtGeom.mjGEOM_HFIELD)

    def test_mujoco_hfield_matches_shared_ground(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('MuJoCo is evaluated locally')
        from deploy_mujoco.terrain import build_model
        scene=Path(__file__).resolve().parents[1]/'resources/robots/go2/mujoco/scene_flat.xml'
        m=build_model(scene,'rolling',3,5,3);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        x,z=profile(3,5,3)
        for a in (1.1,1.5,2.,2.3,3.,4.1,7.):
            geom=np.array([-1],dtype=np.int32)
            dist=mujoco.mj_ray(m,d,np.array([a,0.,1.]),np.array([0.,0.,-1.]),np.array([0,0,0,0,0,1],dtype=np.uint8),1,-1,geom)
            self.assertEqual(geom[0],m.geom('floor').id)
            self.assertAlmostEqual(1-dist,np.interp(a,x,z),places=6)


if __name__=='__main__':unittest.main()
