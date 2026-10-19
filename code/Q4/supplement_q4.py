"""Fixed commitment execution comparisons and non-additive stage VOI diagnostics."""
from q4_utils import *
from q4_features import Factory
from q4_model import Config,summarize
from q4_execution_mpc import mpc_step
from microgrid_online import feedback_step,terminal_value,validate_execution
from q3_model import simulate_paths
def supplement():
    dates,a,fixed,p,records=read_inputs();f=Factory(a,p,records,fixed);v=terminal_value(p[:31].mean(0))
    for kind in (42,43):
        formal=dict(np.load(OUT/f'experiments/q{kind}_formal.npz'))
        comparisons=json.loads((OUT/f'experiments/q{kind}_comparisons.json').read_text())
        for execution in (0,36):
            s={k:x.copy() for k,x in formal.items()};e=6000.
            for day in range(365):
                s['E'][day,0]=e;conf=Config(**json.loads((OUT/f'experiments/q{kind}_calendar.json').read_text())[day])
                for j in range(144):
                    k=j//36 if kind==43 else 0;offset=j-k*36
                    if day==0:action=(0.,0.,max(a[day,j,0]-a[day,j,1],0),max(a[day,j,1]-a[day,j,0],0),e)
                    elif execution:
                        n=min(execution,144-j);pred=formal['predictions'][day,k,offset:offset+n].copy();pred[0]=a[day,j]
                        pp=f.prices.execution_curve(day,j,conf.price_model,formal['price_forecasts'][day,k],k*36,n);pp[0]=p[day,j]
                        # Future G comes from the already issued version, never a later revision.
                        action=mpc_step(formal['versions'][day,k,j:j+n],pred[:,0]-pred[:,1],pp,e,terminal_value(formal['price_forecasts'][day,k,:144]))
                    else:action=feedback_step(formal['G'][day,j],*a[day,j],e)
                    s['C'][day,j],s['D'][day,j],s['H'][day,j],s['W'][day,j],e=action;s['E'][day,j+1]=e
                validate_execution(a[day],{key:value[day] for key,value in s.items()})
            s['emergency_cost']=5*p*s['H'];s['cash']=s['plan_cost']+s['up_cost']+s['penalty']-s['refund']+s['emergency_cost']
            name=f'fixed_commitment_E{execution}';np.savez_compressed(OUT/f'experiments/q{kind}_{name}.npz',**s)
            comparisons[name]=summarize(s,p,a,common_value=v)
        write_json(OUT/f'experiments/q{kind}_comparisons.json',comparisons)
    s=dict(np.load(OUT/'experiments/q43_formal.npz'));rows=[]
    for day in range(31,365):
        for k in (1,2,3):
            t=k*36;initial=s['E'][day,t];old=s['versions'][day,k-1,t:];new=s['versions'][day,k,t:]
            ho,eo=simulate_paths(old,a[day,None,t:],initial);hn,en=simulate_paths(new,a[day,None,t:],initial)
            fee=float((p[day,t:]*(1.5*s['U'][day,k-1,t:]+.5*s['R'][day,k-1,t:])).sum())
            cash=float(5*p[day,t:]@(ho[0]-hn[0]))-fee
            rows.append(dict(day=day,date=str(dates[day]),stage=k,realized_E0_proxy_cash_value=cash,terminal_difference=float(en[0]-eo[0]),adjustment_fee=fee))
    write_csv(OUT/'q43_voi_realized_local.csv',rows)
if __name__=='__main__':supplement()
