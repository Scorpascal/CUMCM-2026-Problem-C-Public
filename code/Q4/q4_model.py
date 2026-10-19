"""Finite policy optimization under causal price information; both official permission sets."""
from dataclasses import dataclass,replace,asdict
import time
import numpy as np
from q4_utils import *
from microgrid_online import feedback_step,validate_execution,terminal_value
from microgrid_adjustment import adjustment_plan,settlement
from q3_model import free_plan,simulate_paths
from q4_execution_mpc import mpc_step
from q4_risk import weighted_quantile,joint_paths,cvar
from q4_price_forecast import MODELS

@dataclass(frozen=True)
class Config:
    price_model:str='mean7'
    q:float=.7
    risk:str='ordinary'
    scenarios:int=8
    alpha:float=.9
    beta:float=0.
    horizon:str='T24'
    execution:int=0 # number of ten-minute lookahead periods; zero is frozen feedback
    @property
    def name(self):
        return f'{self.price_model}_q{self.q:g}_{self.risk}_S{self.scenarios}_a{self.alpha:g}_b{self.beta:g}_{self.horizon}_E{self.execution}'

# Locked before formal rolling optimization. Factorial explosion intentionally avoided.
CANDIDATES=tuple(Config(m,q) for m in MODELS for q in (.65,.8)) + (
    Config(q=0),Config(q=.7),Config(risk='weighted'),Config(q=.8,risk='weighted'),
    Config(risk='joint',scenarios=4),Config(risk='joint'),Config(risk='joint',scenarios=12),
    Config(risk='joint',scenarios=20),Config(risk='joint',beta=.1),
    Config(risk='joint',beta=.2,alpha=.95),Config(horizon='R48'),
    Config(execution=12),Config(execution=36),Config(price_model='ridge30',execution=36))
WARMUP=Config(q=.8)

def horizon(k,conf):return 144-36*k if conf.horizon=='T24' else 288

def risk_net(f,q,n,weighted=False):
    net=f.prediction[:n,0]-f.prediction[:n,1]
    margin=np.zeros(n)
    if q and len(f.residual):
        residual=f.residual[:,:n,0]-f.residual[:,:n,1]
        if weighted:
            weights=np.maximum(.001,f.price_forecast[None,:n]+f.residual[:,:n,2])
            margin=weighted_quantile(residual,weights,q)
        else:margin=np.quantile(residual,q,axis=0)
        margin=np.maximum(0,margin)
    return net+margin,margin

def evaluate(g,paths,price_scenario,initial,value,old=None,original=None,rule='A',beta=0,alpha=.9):
    h,end=simulate_paths(g,paths,initial)
    m=len(g) if old is None else len(old)
    if old is None:fees=price_scenario@g
    else:
        # Forecast cash by scenario; sunk G0 is common and excluded from marginal risk.
        coeff=settlement(old,g[:m],np.ones(m),rule,original)['charge']
        fees=price_scenario[:,:m]@coeff+price_scenario[:,m:]@g[m:]
    cash=fees+(5*price_scenario*h).sum(1)
    score=float(np.mean(cash-value*(end-initial)))+beta*cvar(cash,alpha)
    return score,cash

def choose(f,initial,conf,k,old=None,original=None,rule='A',independent=False,locked_price=None):
    tic=time.perf_counter();m=144-k*36;n=horizon(k,conf)
    assert f.now%144==k*36 and f.latest_end<=f.now and f.source_issue<=f.now
    p=f.price_forecast[:n];value=terminal_value(f.price_forecast[:144])
    if locked_price is not None:p=np.full(n,locked_price)
    if conf.risk=='joint':
        paths,ps,ix=joint_paths(f.prediction,p,f.residual,conf.scenarios,independent)
        if locked_price is not None:ps[:,:m]=locked_price
        qs=(0.,.65,.8,.9)
    else:
        _,margin=risk_net(f,conf.q,n,conf.risk=='weighted')
        path=f.prediction[:n].copy();path[:,0]+=margin
        paths=path[None];ps=p[None];ix=np.array([-1]);qs=(conf.q,)
    choices=[]
    if old is not None:
        full=old.copy()
        if n>m:
            _,end=simulate_paths(old,paths.mean(0)[None,:m],initial)
            net,_=risk_net(f,conf.q,n,conf.risk=='weighted')
            full=np.r_[old,free_plan(net[m:],p[m:],float(end[0]),value)['G']]
        score,cash=evaluate(full,paths,ps,initial,value,old,original,rule,conf.beta,conf.alpha)
        choices.append((full,score,'KEEP'))
    for q in qs:
        net,_=risk_net(f,q,n,conf.risk=='weighted')
        sol=free_plan(net,p,initial,value) if old is None else adjustment_plan(net,p,initial,old,value,rule,original)
        score,cash=evaluate(sol['G'],paths,ps,initial,value,old,original,rule,conf.beta,conf.alpha)
        choices.append((sol['G'],score,f'q={q:g}'))
    best=min(range(len(choices)),key=lambda i:(choices[i][1],i))
    keep=choices[0][1] if old is not None else np.nan
    proposal=min(choices[1:],key=lambda v:v[1]) if old is not None else choices[best]
    voi=keep-proposal[1] if old is not None else np.nan
    if old is not None and voi<=1e-6:best=0
    g,score,label=choices[best]
    return g[:m].copy(),dict(voi=voi,keep_score=keep,update_score=proposal[1],choice=label,
        accepted=old is not None and np.max(np.abs(g[:m]-old))>1e-7,seconds=time.perf_counter()-tic,
        now=f.now,source_issue=f.source_issue,residual_latest_end=f.latest_end,scenario_count=len(paths),score=score)

def run_day(factory,kind,day,initial,conf,updates=(1,2,3),stale_price=False,stale_pv=False,
            oracle='',rule='A',locked=False,independent=False,current_known=True,pv0=False):
    tic=time.perf_counter();a=factory.actual[day];realized=factory.prices.realized[day]
    s={k:np.zeros(144) for k in ('G','C','D','H','W','plan_cost','up_cost','penalty','refund','emergency_cost')}
    s['E']=np.zeros(145);s['E'][0]=initial
    versions=np.zeros((4,144));u=np.zeros((3,144));r=u.copy();g=np.zeros(144)
    price_versions=np.full((4,288),np.nan);net_versions=np.full((4,288,2),np.nan)
    events=[];last_features=None;last_start=0
    for k in range(4):
        t=k*36;previous=g.copy()
        active=k==0 or (kind==43 and k in updates)
        if active:
            fk=43 if pv0 and kind==42 else kind
            f=factory.feature(fk,day,k,conf.price_model,stale_price,stale_pv)
            if oracle:
                pred=f.prediction.copy();price=f.price_forecast.copy();n=min(288,len(factory.flat)-f.now)
                residual=f.residual.copy()
                # Oracle-P knows today's full prices only, cross-day tail remains causal.
                if oracle in ('P','FULL'):
                    m=144-t;price[:m]=realized[t:]
                    residual[:,:m,2]=0 # revealed price carries no artificial scenario error
                if oracle in ('NP','FULL'):
                    pred[:n]=factory.flat[f.now:f.now+n]
                    residual[:,:n,:2]=0
                f=replace(f,prediction=pred,price_forecast=price,residual=residual)
            last_features=f;last_start=t
            price_versions[k]=f.price_forecast;net_versions[k]=f.prediction
            issue_price=float(factory.prices.flat[f.now-1]) if f.now else .5
            if day==0:
                decision=dict(voi=0.,keep_score=0.,update_score=0.,choice='startup',accepted=False,seconds=0.,now=f.now,source_issue=f.source_issue,residual_latest_end=f.latest_end,scenario_count=0,score=0.)
            else:
                g[t:],decision=choose(f,float(s['E'][t]),conf,k,None if k==0 else g[t:],None if k==0 else versions[0,t:],
                    rule,independent,issue_price if locked else None)
            if k==0:
                s['plan_cost']=g*(issue_price if locked else realized)
            else:
                price_bill=np.full(144-t,issue_price) if locked else realized[t:]
                bill=settlement(previous[t:],g[t:],price_bill,rule,versions[0,t:])
                u[k-1,t:]=bill['U'];r[k-1,t:]=bill['R']
                for key in ('up_cost','penalty','refund'):s[key][t:]+=bill[key]
                events.append(dict(day=day,stage=k,**decision,initial=float(s['E'][t]),increase=float(bill['U'].sum()),
                    decrease=float(bill['R'].sum()),realized_adjustment=float(bill['charge'].sum())))
        versions[k]=g
        assert np.array_equal(g[:t],previous[:t])
        for j in range(t,t+36):
            s['G'][j]=g[j];e=s['E'][j]
            if day==0:
                rr=g[j]+a[j,1]-a[j,0];action=(0.,0.,max(-rr,0),max(rr,0),e)
            elif conf.execution:
                n=min(conf.execution,144-j);offset=j-last_start
                pred=last_features.prediction[offset:offset+n].copy()
                pred[0]=a[j] # current aggregate, never later actuals
                pp=factory.prices.execution_curve(day,j,conf.price_model,last_features.price_forecast,last_start,n)
                pp[0]=realized[j] if current_known else (factory.prices.flat[day*144+j-1] if day*144+j else .5)
                value=terminal_value(last_features.price_forecast[:144])
                action=mpc_step(g[j:j+n],pred[:,0]-pred[:,1],pp,e,value)
            else:action=feedback_step(g[j],*a[j],e)
            s['C'][j],s['D'][j],s['H'][j],s['W'][j],s['E'][j+1]=action
    validate_execution(a,s)
    s['emergency_cost']=5*realized*s['H'];s['cash']=s['plan_cost']+s['up_cost']+s['penalty']-s['refund']+s['emergency_cost']
    s.update(versions=versions,U=u,R=r,price_forecasts=price_versions,predictions=net_versions,
             config=conf.name,seconds=time.perf_counter()-tic)
    return s,events

def stack(days):return {k:np.array([d[k] for d in days]) for k in days[0]}

def summarize(s,realized,actual,start=31,common_value=.5):
    x={k:v[start:] for k,v in s.items()};p=realized[start:];a=actual[start:];cash=x['cash'].sum(1);e=x['E'];h=x['H']
    high=p>=np.quantile(p,.9);low=p<=np.quantile(p,.1);g=x['G'];w=x['W']
    r={k+'_yuan':float(x[k].sum()) for k in ('plan_cost','up_cost','penalty','refund','emergency_cost')}
    r.update(total_cost_yuan=float(cash.sum()),daily_mean_yuan=float(cash.mean()),daily_p95_yuan=float(np.quantile(cash,.95)),
        worst5_mean_yuan=float(np.sort(cash)[-int(np.ceil(.05*len(cash))):].mean()),max_daily_yuan=float(cash.max()),
        terminal_adjusted_cost_yuan=float(cash.sum()-common_value*(e[-1,-1]-e[0,0])),
        initial_kwh=float(e[0,0]),terminal_kwh=float(e[-1,-1]),energy_min=float(e.min()),energy_max=float(e.max()),
        plan_kwh=float(x['versions'][:,0].sum()),effective_kwh=float(g.sum()),emergency_kwh=float(h.sum()),
        emergency_periods=int((h>1e-7).sum()),emergency_days=int(np.any(h>1e-7,1).sum()),max_emergency_kwh=float(h.max()),
        high_price_emergency_kwh=float(h[high].sum()),increase_kwh=float(x['U'].sum()),decrease_kwh=float(x['R'].sum()),
        adjustment_events=int(np.any(x['U']+x['R']>1e-7,2).sum()),charge_kwh=float(x['C'].sum()),discharge_kwh=float(x['D'].sum()),
        waste_kwh=float(w.sum()),pv_curtail_lower_kwh=float(np.maximum(w-g,0).sum()),pv_curtail_upper_kwh=float(np.minimum(w,a[:,:,1]).sum()),
        pv_utilization_lower=float(1-np.minimum(w,a[:,:,1]).sum()/a[:,:,1].sum()),
        pv_utilization_upper=float(1-np.maximum(w-g,0).sum()/a[:,:,1].sum()),
        lower_bound_states=int((e<=1200+1e-7).sum()),upper_bound_states=int((e>=10800-1e-7).sum()),
        continuity_error=float(np.max(np.abs(e[1:,0]-e[:-1,-1]))),
        low_price_charge_share=float(x['C'][low].sum()/max(x['C'].sum(),1e-9)),
        high_price_discharge_share=float(x['D'][high].sum()/max(x['D'].sum(),1e-9)),
        grid_weighted_price=float((p*(g+h)).sum()/max((g+h).sum(),1e-9)),
        high_price_grid_share=float((g+h)[high].sum()/max((g+h).sum(),1e-9)),
        execution_seconds=float(x['seconds'].sum()) if 'seconds' in x else 0.,days=len(cash))
    return r
