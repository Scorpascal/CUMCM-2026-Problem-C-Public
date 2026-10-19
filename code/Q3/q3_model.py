"""预注册策略类、当前信息VOI与严格逐时环境。"""
from dataclasses import dataclass,replace,asdict
from datetime import datetime,timedelta
import time
import numpy as np
from microgrid_common import Battery
from microgrid_online import plan_net,feedback_step,validate_execution,terminal_value
from microgrid_adjustment import settlement,adjustment_plan
from q3_forecast import risk_net,scenarios

@dataclass(frozen=True)
class Config:
    interp:str='linear'
    q:float=.8
    horizon:str='T24'
    gain:float=0.
    bias:bool=False
    scenario:bool=False
    gate:bool=True
    @property
    def name(self):
        return f'{self.interp}_q{self.q:.2f}_{self.horizon}_g{self.gain:g}_b{int(self.bias)}_s{int(self.scenario)}_gate{int(self.gate)}'

CANDIDATES=tuple(Config(q=q) for q in (0.,.65,.7,.8,.85,.9))+(
    Config(interp='constant'),Config(interp='pchip'),Config(bias=True),
    Config(gain=.5),Config(gain=1.),Config(horizon='R24'),Config(horizon='R48'),
    Config(scenario=True),Config(gate=False),Config(scenario=True,horizon='R48'))
WARMUP=Config()

def horizon_size(k,conf):
    return 144-36*k if conf.horizon=='T24' else (144 if conf.horizon=='R24' else 288)

def simulate_paths(g,paths,initial,battery=Battery()):
    """向量化多个路径，但各路径的动作只用当前观测和自身E；没有自由场景储能变量。"""
    paths=np.asarray(paths);s,n,_=paths.shape;b=battery
    e=np.full(s,float(initial));h=np.zeros((s,n))
    for t in range(n):
        r=g[t]+paths[:,t,1]-paths[:,t,0]
        c=np.minimum(np.minimum(np.maximum(r,0),b.power*b.dt),np.maximum(0,(b.maximum-e)/b.eta_charge))
        d=np.minimum(np.minimum(np.maximum(-r,0),b.power*b.dt),np.maximum(0,(e-b.minimum)*b.eta_discharge))
        h[:,t]=np.maximum(0,-r-d);e+=b.eta_charge*c-d/b.eta_discharge
    return h,e

def choose_stage(f,initial,price,conf,k,old=None,original=None,rule='A',threshold=1e-6,battery=Battery()):
    """唯一决策接口只收到当前StageFeatures；不能访问环境或未来发行版本。"""
    tic=time.perf_counter();t=k*36;m=144-t;n=horizon_size(k,conf)
    assert f.now%144==t and f.source_issue<=f.now and f.residual_latest_end<=f.now
    p=np.tile(price,3)[t:t+n];value=terminal_value(price,battery)
    qs=(0.,.65,.8,.9) if conf.scenario else (conf.q,)
    # 同一个当前信息下的相同评价路径用于所有KEEP/UPDATE候选。
    if conf.scenario:paths=scenarios(f,n)
    else:
        _,margin=risk_net(f,conf.q,n)
        path=f.prediction[:n].copy();path[:,0]+=margin;paths=path[None]
    nominal=paths.mean(axis=0)
    tailnet=nominal[m:,0]-nominal[m:,1]

    def evaluate(g):
        g=np.asarray(g[:m]).copy()
        # 前瞻购电只是一条当前信息生成的辅助策略，不进入实际承诺/账本。
        h0,e0=simulate_paths(g,nominal[None,:m],initial,battery)
        if n>m:
            # 未来规划使用固定全日电量价值，通过专用无承诺函数。
            future=free_plan(tailnet,p[m:],float(e0[0]),value,battery)
            full=np.r_[g,future['G']]
        else:full=g
        h,e=simulate_paths(full,paths,initial,battery)
        fee=float(p[:m]@g) if old is None else float(settlement(old,g,p[:m],rule,original)['charge'].sum())
        scores=fee+float(p[m:]@full[m:])+h@(5*p)-value*(e-initial)
        return float(scores.mean())

    choices=[]
    if old is not None:choices.append((old.copy(),evaluate(old),'KEEP'))
    for q in qs:
        net,margin=risk_net(f,q,n)
        sol=free_plan(net,p,initial,value,battery) if old is None else adjustment_plan(net,p,initial,old,value,rule,original,battery)
        g=sol['G'][:m];choices.append((g,evaluate(g),f'q={q}'))
    # 稳定平分优先KEEP；无门控候选强制使用提出的方案。
    start=1 if old is not None and not conf.gate else 0
    best=min(range(start,len(choices)),key=lambda i:(choices[i][1],i))
    keep=choices[0][1] if old is not None else np.nan
    proposed=min(choices[1:],key=lambda v:v[1]) if old is not None else choices[best]
    voi=keep-proposed[1] if old is not None else np.nan
    if old is not None and conf.gate and voi<=threshold:best=0
    g,score,label=choices[best]
    return dict(G=g.copy(),proposed=proposed[0].copy(),keep_score=keep,update_score=proposed[1],voi=voi,
                accepted=old is not None and bool(np.max(np.abs(g-old))>1e-7),
                choice=label,seconds=time.perf_counter()-tic,source_issue=f.source_issue,
                residual_latest_end=f.residual_latest_end,score=score)

def free_plan(net,p,initial,value,battery=Battery()):
    # 与Q2普通计划LP相同，用显式value避免截断价格改变终端代理价值。
    from scipy.optimize import linprog
    from microgrid_online import balance_matrix
    n=len(net);b=battery;obj=np.zeros(5*n+1);obj[:n]=p;obj[-1]=-value
    lo=np.zeros(5*n+1);hi=np.full(5*n+1,np.inf)
    hi[n:3*n]=b.power*b.dt;lo[4*n:]=b.minimum;hi[4*n:]=b.maximum
    lo[4*n]=hi[4*n]=np.clip(initial,b.minimum,b.maximum)
    res=linprog(obj,A_eq=balance_matrix(n,b.eta_charge,b.eta_discharge),b_eq=np.r_[net,np.zeros(n)],bounds=np.column_stack([lo,hi]),method='highs')
    if not res.success:raise RuntimeError(res.message)
    return dict(G=res.x[:n].copy(),E=res.x[4*n:].copy(),objective=float(res.fun))

def run_day(factory,day,initial,price,conf,updates=(1,2,3),rule='A',stale=False,battery=Battery(),details=False):
    """环境逐段揭示真实值；控制器看到的特征由factory严格截断。"""
    actual=factory.actual[day];g=np.zeros(144);versions=np.zeros((4,144))
    s={key:np.zeros(144) for key in ('G','C','D','H','W')};s['E']=np.zeros(145);s['E'][0]=initial
    fees={key:np.zeros(144) for key in ('up_cost','penalty','refund')}
    u=np.zeros((3,144));r=u.copy();pred=np.full((4,144,2),np.nan);events=[];latest=0
    for k in range(4):
        t=36*k;e=float(s['E'][t]);old=g.copy()
        if k==0 or k in updates:
            if stale is True:latest=0
            elif not isinstance(stale,tuple) or k not in stale:latest=k
            f=factory.feature(day,k,conf.interp,conf.gain,conf.bias,latest)
            pred[k,t:]=f.prediction[:144-t]
            if day==0:
                decision=dict(G=np.zeros(144-t),proposed=np.zeros(144-t),keep_score=0.,update_score=0.,voi=0.,accepted=False,choice='startup',seconds=0.,source_issue=f.source_issue,residual_latest_end=f.residual_latest_end)
            else:decision=choose_stage(f,e,price,conf,k,None if k==0 else old[t:],None if k==0 else versions[0,t:],rule,battery=battery)
            g[t:]=decision['G']
            if k:
                bill=settlement(old[t:],g[t:],price[t:],rule,versions[0,t:])
                u[k-1,t:]=bill['U'];r[k-1,t:]=bill['R']
                for key in fees:fees[key][t:]+=bill[key]
                ev={key:decision[key] for key in ('keep_score','update_score','voi','accepted','choice','seconds','source_issue','residual_latest_end')}
                ev.update(day=day,stage=k,initial=e,increase_kwh=float(bill['U'].sum()),decrease_kwh=float(bill['R'].sum()),adjustment_charge=float(bill['charge'].sum()))
                if details:
                    # 事后固定余日对照，只作诊断；在决策已完成之后读取实际未来，绝不返回控制器。
                    hk,ek=simulate_paths(old[t:],actual[None,t:],e,battery)
                    hu,eu=simulate_paths(g[t:],actual[None,t:],e,battery)
                    ev['realized_value']=float(5*price[t:]@(hk[0]-hu[0])-bill['charge'].sum()+terminal_value(price,battery)*(eu[0]-ek[0]))
                events.append(ev)
        versions[k]=g
        assert np.array_equal(g[:t],old[:t])
        for j in range(t,t+36):
            s['G'][j]=g[j]
            if day==0:
                rr=g[j]+actual[j,1]-actual[j,0]
                s['H'][j]=max(-rr,0);s['W'][j]=max(rr,0);s['E'][j+1]=s['E'][j]
            else:
                s['C'][j],s['D'][j],s['H'][j],s['W'][j],s['E'][j+1]=feedback_step(g[j],*actual[j],s['E'][j],battery)
    validate_execution(actual,s,battery)
    s.update(versions=versions,U=u,R=r,prediction=pred,**fees)
    s['plan_cost']=versions[0]*price;s['emergency_cost']=5*price*s['H']
    s['cash']=s['plan_cost']+s['up_cost']+s['penalty']-s['refund']+s['emergency_cost']
    s['config']=conf.name
    return s,events

def stack_days(days):
    return {key:np.asarray([s[key] for s in days]) for key in days[0]}

def summarize(s,price,start=31):
    x={k:v[start:] for k,v in s.items()};e=x['E'];g=x['G'];h=x['H'];cash=x['cash'].sum(axis=1)
    result={k+'_yuan':float(x[k].sum()) for k in ('plan_cost','up_cost','penalty','refund','emergency_cost')}
    result.update(total_cost_yuan=float(cash.sum()),terminal_adjusted_cost_yuan=float(cash.sum()-terminal_value(price)*(e[-1,-1]-e[0,0])),
        days=len(e),initial_kwh=float(e[0,0]),terminal_kwh=float(e[-1,-1]),emergency_kwh=float(h.sum()),
        emergency_periods=int((h>1e-7).sum()),emergency_days=int(np.any(h>1e-7,axis=1).sum()),max_emergency_kwh=float(h.max()),
        increase_kwh=float(x['U'].sum()),decrease_kwh=float(x['R'].sum()),plan_kwh=float(x['versions'][:,0].sum()),effective_kwh=float(g.sum()),
        adjustment_events=int(np.any(x['U']+x['R']>1e-7,axis=2).sum()),adjustment_days=int(np.any(x['U']+x['R']>1e-7,axis=(1,2)).sum()),
        daily_p95_yuan=float(np.quantile(cash,.95)),energy_min=float(e.min()),energy_max=float(e.max()),
        charge_kwh=float(x['C'].sum()),discharge_kwh=float(x['D'].sum()),waste_kwh=float(x['W'].sum()),
        lower_bound_states=int((e<=1200+1e-6).sum()),upper_bound_states=int((e>=10800-1e-6).sum()),
        continuity_error=float(np.max(np.abs(e[1:,0]-e[:-1,-1]))) if len(e)>1 else 0.)
    return result
