"""预先注册的候选集合；计划层完全不接收当天真实数据。"""
from dataclasses import dataclass,asdict
import numpy as np
from microgrid_common import Battery
from microgrid_online import plan_net,execute_day,terminal_value
from forecast_q2 import METHODS,risk_net

QUANTILES=(0.,.65,.70,.75,.80,.85,.90,.95)
WINDOWS=((1,'cycle'),(1,'value'),(2,'cycle'),(3,'cycle'))


@dataclass(frozen=True)
class Config:
    method:str='blend'
    q:float=.8
    horizon:int=2
    terminal:str='cycle'
    scenario:bool=False

    @property
    def name(self):
        return 'scenario_blend_48' if self.scenario else f'{self.method}_q{self.q:.2f}_{24*self.horizon}h_{self.terminal}'


CANDIDATES=tuple(Config(m,q,h,t) for m in METHODS for q in QUANTILES for h,t in WINDOWS)+(Config(scenario=True),)
WARMUP=Config('mean7',.8,2,'cycle')


def make_plan(features,initial,price,config,battery=Battery()):
    assert features.training_max<features.day
    if config.scenario:
        return scenario_plan(features,initial,price,battery)
    net,margin=risk_net(features,config.method,config.q,config.horizon)
    result=plan_net(net,np.tile(price,config.horizon),initial,config.terminal,battery)
    result.update(q=config.q,margin=margin[0],prediction=features.prediction[config.method][0],
                  config=config.name,net=net)
    return result


def scenario_plan(features,initial,price,battery=Battery()):
    """有限策略SAA；每条场景按同一feedback_step执行，无场景全知储能。"""
    r=features.residual['blend'][0]
    if not len(r):
        return make_plan(features,initial,price,Config(),battery)
    pred=features.prediction['blend'][0]
    scenes=np.maximum(pred[None,:,:]+r,0)
    # 至多12条按时间均匀保留的完整日轨迹；没有逐时点独立抽样。
    scenes=scenes[np.unique(np.linspace(0,len(scenes)-1,min(12,len(scenes))).astype(int))]
    value=terminal_value(price,battery)
    best=None;best_score=np.inf;seconds=0
    for q in QUANTILES[1:]:
        plan=make_plan(features,initial,price,Config('blend',q,2,'cycle'),battery)
        seconds+=plan['seconds']
        scores=[]
        for scene in scenes:
            s=execute_day(plan['G'][:144],scene,initial,battery)
            scores.append(float(price@s['G']+5*price@s['H']-value*(s['E'][-1]-initial)))
        score=float(np.mean(scores))
        if score<best_score-1e-8:
            best,best_score=plan,score
    best.update(config='scenario_blend_48',scenario_score=best_score,seconds=seconds)
    return best


def select_config(features,past_actual,known_energy,start_day,price,battery=Battery(),candidates=CANDIDATES):
    """月首只用过去21日，验证段共同起点、各候选独立连续E，终点价值调整。"""
    assert len(past_actual)==start_day and len(features)>=start_day
    begin=max(1,start_day-21)
    initial=float(known_energy[begin])
    value=terminal_value(price,battery)
    rows=[]
    for conf in candidates:
        e=initial;cash=0.;emergency=0.;seconds=0.
        for k in range(begin,start_day):
            plan=make_plan(features[k],e,price,conf,battery)
            real=execute_day(plan['G'][:144],past_actual[k],e,battery)
            cash+=float(price@real['G']+5*price@real['H'])
            emergency+=float(real['H'].sum());seconds+=plan['seconds'];e=real['E'][-1]
        rows.append(dict(selection_day=start_day,begin=begin,end=start_day-1,config=conf.name,
                         cash_cost=cash,score=cash-value*(e-initial),initial=initial,terminal=float(e),
                         emergency_kwh=emergency,solve_seconds=seconds))
    idx=min(range(len(rows)),key=lambda i:(rows[i]['score'],i))
    for i,row in enumerate(rows):row['selected']=i==idx
    return candidates[idx],rows
