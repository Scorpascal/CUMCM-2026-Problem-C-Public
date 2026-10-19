"""Independent counterfactual, synthetic physics, permission and cash tests."""
import unittest
from dataclasses import replace
from datetime import datetime,timedelta
import numpy as np
from q4_utils import *
from q4_features import Factory
from q4_price_forecast import daily_price,PriceFactory,MODELS
from q4_model import Config,run_day,choose,evaluate,CANDIDATES
from q4_risk import cvar,weighted_quantile,joint_paths
from q4_execution_mpc import mpc_step
from microgrid_online import feedback_step,validate_execution
from microgrid_adjustment import settlement,adjustment_plan

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates,cls.a,cls.fixed,cls.p,cls.records=read_inputs();cls.f=Factory(cls.a,cls.p,cls.records,cls.fixed)

    def test_01_protected(self):self.assertEqual(check_protected(),0)
    def test_02_calendar(self):
        self.assertEqual(self.p.shape,(365,144));self.assertEqual(self.dates[-1]-self.dates[0],timedelta(days=364))
    def test_03_price_prefix(self):
        for model in MODELS:
            p=self.p.copy();p[31:]*=10
            x=self.f.prices.raw(31,0,model);y=PriceFactory(p).raw(31,0,model)
            np.testing.assert_array_equal(x,y)
    def test_04_reject_uncut_history(self):
        with self.assertRaises(AssertionError):daily_price(self.p,31,'mean7')
    def test_05_constant_price(self):
        for model in MODELS:
            z=daily_price(np.ones((31,144))*.8,31,model)
            np.testing.assert_allclose(z,.8,atol=1e-7)
    def test_06_no_history(self):
        for model in MODELS:self.assertTrue(np.isfinite(daily_price(np.empty((0,144)),0,model)).all())
    def test_07_joint_sync(self):
        for kind in (42,43):
            f=self.f.feature(kind,60,2,'mean7')
            for j,d in enumerate(f.origins):
                now=d*144+72
                np.testing.assert_allclose(f.residual[j,:,:2],self.f.flat[now:now+288]-self.f.raw_net(kind,d,2,2))
                np.testing.assert_allclose(f.residual[j,:,2],self.p.ravel()[now:now+288]-self.f.prices.raw(d,2,'mean7'))
                self.assertLessEqual(now+288,f.now)
    def test_08_scenarios_synchronized_indices(self):
        f=self.f.feature(43,60,1,'mean7');a,p,ix=joint_paths(f.prediction,f.price_forecast[:108],f.residual)
        np.testing.assert_allclose(p,np.maximum(.001,f.price_forecast[None,:108]+f.residual[ix,:108,2]))
    def test_09_price_future_G0(self):
        p=self.p.copy();p[60:]*=3;ff=Factory(self.a,p,self.records,self.fixed)
        for kind in (42,43):
            for conf in (Config(),Config(risk='weighted'),Config(risk='joint')):
                x,_=run_day(self.f,kind,60,6000,conf);y,_=run_day(ff,kind,60,6000,conf)
                np.testing.assert_array_equal(x['versions'][0],y['versions'][0])
                self.assertGreater(abs(x['cash'].sum()-y['cash'].sum()),1)
    def test_10_stage_price_future(self):
        for t in (72,108):
            p=self.p.copy();p[60,t:]*=2;p[61:]*=2
            x,_=run_day(self.f,43,60,6000,Config());y,_=run_day(Factory(self.a,p,self.records,self.fixed),43,60,6000,Config())
            np.testing.assert_array_equal(x['versions'][:t//36+1],y['versions'][:t//36+1])
    def test_11_future_actual(self):
        a=self.a.copy();a[60:,:,0]*=3
        for kind in (42,43):
            x,_=run_day(self.f,kind,60,6000,Config());y,_=run_day(Factory(a,self.p,self.records,self.fixed),kind,60,6000,Config())
            np.testing.assert_array_equal(x['versions'][0],y['versions'][0]);self.assertGreater(abs(x['H'].sum()-y['H'].sum()),1)
    def test_12_future_pv_release(self):
        rec=dict(self.records)
        for k in (2,3):
            dt=datetime(2025,1,1)+timedelta(days=60,hours=k*6)
            rec[dt]=replace(rec[dt],power=tuple(np.array(rec[dt].power)*.1))
        x,_=run_day(self.f,43,60,6000,Config());y,_=run_day(Factory(self.a,self.p,rec,self.fixed),43,60,6000,Config())
        np.testing.assert_array_equal(x['versions'][:2],y['versions'][:2])
    def test_13_q42_has_no_pv_permissions(self):
        rec={dt:replace(v,power=tuple(np.zeros(24))) for dt,v in self.records.items()}
        x,_=run_day(self.f,42,60,6000,Config());y,_=run_day(Factory(self.a,self.p,rec,self.fixed),42,60,6000,Config())
        np.testing.assert_array_equal(x['G'],y['G'])
    def test_14_q42_fixed_G(self):
        x,ev=run_day(self.f,42,60,6000,Config(execution=12))
        for k in range(4):np.testing.assert_array_equal(x['versions'][k],x['versions'][0])
        self.assertEqual(len(ev),0);self.assertEqual(x['U'].sum()+x['R'].sum(),0)
    def test_15_q43_nonretroactivity(self):
        x,_=run_day(self.f,43,60,6000,Config())
        for k in (1,2,3):
            np.testing.assert_array_equal(x['versions'][k,:k*36],x['versions'][k-1,:k*36])
            np.testing.assert_allclose(x['versions'][k],x['versions'][k-1]+x['U'][k-1]-x['R'][k-1],atol=1e-8)
    def test_16_dynamic_settlement_hand(self):
        p=np.array([.2,.7,1.4]);g=np.array([100.,80.,50.]);total=p@g;up=down=0.
        for new in (np.array([120.,90.,60.]),np.array([110.,80.,50.]),np.array([120.,100.,70.])):
            b=settlement(g,new,p);up+=b['up_cost'].sum();down+=b['penalty'].sum();g=new
        # U totals 30,30,30; R 10,10,10; emergency 1,2,3.
        self.assertAlmostEqual(up,1.5*30*p.sum());self.assertAlmostEqual(down,.5*10*p.sum())
        self.assertAlmostEqual(total+up+down+5*p@np.array([1.,2.,3.]),146+103.5+11.5+29)
    def test_17_multiple_adjustments(self):
        for rule,expected in [('A',165),('B',125),('B_net',95)]:
            old=np.array([100.]);cost=100
            for x in (80,110,90):
                new=np.array([float(x)]);cost+=settlement(old,new,np.ones(1),rule,np.array([100.]))['charge'].sum();old=new
            self.assertAlmostEqual(cost,expected)
    def test_18_cvar_lp_identity(self):
        from scipy.optimize import linprog
        x=np.array([1.,2.,10.,30.]);a=.9
        sol=linprog(np.r_[1.,np.full(4,1/(4*(1-a)))],A_ub=np.column_stack([-np.ones(4),-np.eye(4)]),b_ub=-x,bounds=[(None,None)]+[(0,None)]*4)
        self.assertAlmostEqual(cvar(x,a),sol.fun)
    def test_19_cvar_beta_zero(self):
        paths=np.array([[[10.,0.],[20.,0.]],[[40.,0.],[50.,0.]]]);p=np.ones((2,2));g=np.zeros(2)
        score,cash=evaluate(g,paths,p,1200,0,beta=0);self.assertAlmostEqual(score,cash.mean())
    def test_20_price_weighted_tail(self):
        v=np.arange(10.).reshape(-1,1);w=np.ones_like(v);w[-1]=20
        self.assertGreater(weighted_quantile(v,w,.8)[0],weighted_quantile(v,np.ones_like(v),.8)[0])
    def test_21_mpc_low_now_high_later(self):
        x=mpc_step(np.zeros(2),np.full(2,500.),np.array([.01,2.]),1200+500/.9,0)
        self.assertLess(x[1],1e-6);self.assertAlmostEqual(x[2],500.)
    def test_22_mpc_no_emergency_charging(self):
        x=mpc_step(np.zeros(2),np.array([500.,500.]),np.array([.01,2.]),6000,0)
        self.assertEqual(x[0],0.)
        y=mpc_step(np.array([100.,0.]),np.array([0.,500.]),np.array([.01,2.]),6000,0)
        self.assertEqual(y[2],0.);self.assertAlmostEqual(y[0],100)
    def test_23_mpc_physics_wrong_forecast(self):
        x,_=run_day(self.f,43,60,6000,Config(price_model='last',execution=12));validate_execution(self.a[60],x)
    def test_24_mpc_future_prefix(self):
        p=self.p.copy();p[60,72:]*=10
        x,_=run_day(self.f,42,60,6000,Config(execution=12));y,_=run_day(Factory(self.a,p,self.records,self.fixed),42,60,6000,Config(execution=12))
        for key in ('G','C','D','H','W'):np.testing.assert_array_equal(x[key][:72],y[key][:72])
    def test_25_price_scale(self):
        f=self.f.feature(42,60,0,'mean7');scaled=replace(f,price_forecast=f.price_forecast*2,residual=f.residual*np.array([1,1,2]))
        x,_=choose(f,6000,Config(),0);y,_=choose(scaled,6000,Config(),0)
        np.testing.assert_allclose(x,y,atol=1e-6)
    def test_26_month_future_independent(self):
        from solve_q4 import select_config
        p=self.p.copy();p[31:]*=10
        configs=(Config(),Config(price_model='last'))
        x,rx=select_config(self.f,42,31,[6000]*32,configs)
        y,ry=select_config(Factory(self.a,p,self.records,self.fixed),42,31,[6000]*32,configs)
        self.assertEqual(x,y)
        np.testing.assert_allclose([r['score'] for r in rx],[r['score'] for r in ry],rtol=0,atol=1e-8)
    def test_27_adjustment_lp_milp(self):
        f=self.f.feature(43,60,3,'mean7');net=f.prediction[:36,0]-f.prediction[:36,1];old=np.full(36,400.)
        x=adjustment_plan(net,f.price_forecast[:36],6000,old,.5)
        y=adjustment_plan(net,f.price_forecast[:36],6000,old,.5,integer=True)
        self.assertAlmostEqual(x['objective'],y['objective'],places=5)

class OutputTests(unittest.TestCase):
    def check(self,kind):
        if not (OUT/f'result4-{kind-40}.xlsx').exists():self.skipTest('Workbook not yet generated; mandatory final audit checks it')
        from audit_q4 import audit_kind
        dates,a,fixed,p,records=read_inputs();audit_kind(kind,dates,a,p)
    def test_28_q42_full_output(self):self.check(42)
    def test_29_q43_full_output(self):self.check(43)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests),unittest.defaultTestLoader.loadTestsFromTestCase(OutputTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
