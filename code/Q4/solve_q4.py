"""Reproduce Q4: preparation, prechecks, causal selection, diagnostics and deliverables."""
import argparse,subprocess,time
from dataclasses import asdict,replace
from datetime import datetime,timedelta
import numpy as np
from q4_utils import *
from q4_features import Factory
from q4_model import *

def select_config(factory,kind,day,known,candidates=CANDIDATES,window=21):
    begin=max(1,day-window);start=float(known[begin]);rows=[]
    # Explicitly remove future actuals, prices and official forecast releases.
    cutoff=datetime(2025,1,1)+timedelta(days=day)
    past=Factory(factory.actual[:day],factory.prices.realized[:day],{d:r for d,r in factory.records.items() if d<cutoff},factory.fixed)
    value=terminal_value(factory.prices.realized[max(0,day-7):day].mean(0))
    for conf in candidates:
        e=start;cash=0.;tic=time.perf_counter()
        for d in range(begin,day):
            s,_=run_day(past,kind,d,e,conf);cash+=s['cash'].sum();e=s['E'][-1]
        rows.append(dict(kind=kind,selection_day=day,begin=begin,end=day-1,config=conf.name,cash=float(cash),
            score=float(cash-value*(e-start)),initial=start,terminal=float(e),value=value,seconds=time.perf_counter()-tic))
    i=min(range(len(rows)),key=lambda j:(rows[j]['score'],j))
    for j,r in enumerate(rows):r['selected']=j==i
    # Drop method cache keys referring to this bounded validation environment.
    Factory.feature.cache_clear();Factory.raw_net.cache_clear()
    return candidates[i],rows

def replay(factory,kind,calendar=None,conf=None,**kwargs):
    e=6000.;days=[];events=[]
    for d in range(len(factory.actual)):
        c=WARMUP if d<31 else (Config(**calendar[d]) if calendar else conf)
        s,ev=run_day(factory,kind,d,e,c,**kwargs);e=s['E'][-1];days.append(s);events.extend(ev)
    return stack(days),events

def compute_kind(kind):
    writable();assert (OUT/'experiments/core_tests_passed.json').exists()
    dates,a,fixed,p,records=read_inputs();f=Factory(a,p,records,fixed)
    common_value=terminal_value(p[:31].mean(0));e=6000.;known=[e];days=[];calendar=[];selection=[];events=[];conf=WARMUP
    tic=time.perf_counter()
    for d,date in enumerate(dates):
        if d>=31 and date.day==1:
            conf,rows=select_config(f,kind,d,known);selection.extend(rows)
            write_csv(OUT/f'experiments/q{kind}_monthly_selection.csv',selection)
            print('Q',kind,date,conf.name,flush=True)
        s,ev=run_day(f,kind,d,e,conf);e=float(s['E'][-1]);known.append(e);days.append(s);calendar.append(asdict(conf));events.extend(ev)
    result=stack(days)
    np.savez_compressed(OUT/f'experiments/q{kind}_formal.npz',**result)
    write_json(OUT/f'experiments/q{kind}_calendar.json',calendar)
    write_csv(OUT/f'q{kind}_voi_events.csv',events,fields=['kind'] if not events else None)
    write_json(OUT/f'q{kind}_summary.json',summarize(result,p,a,common_value=common_value))
    write_json(OUT/f'experiments/q{kind}_formal_complete.json',dict(seconds=time.perf_counter()-tic,completed_utc=datetime.now().isoformat(),fixed_rankings_not_yet_used=True))
    print('Q',kind,'formal complete',time.perf_counter()-tic,flush=True)

def diagnostics_kind(kind):
    writable();dates,a,fixed,p,records=read_inputs();f=Factory(a,p,records,fixed)
    cal=json.loads((OUT/f'experiments/q{kind}_calendar.json').read_text());value=terminal_value(p[:31].mean(0))
    summary=[];daily=[]
    # Restartable diagnostics never affect the already selected formal calendar.
    for conf in CANDIDATES:
        file=OUT/f'experiments/q{kind}_candidate_{CANDIDATES.index(conf)}.npz'
        if file.exists():s=dict(np.load(file))
        else:
            s,_=replay(f,kind,conf=conf);np.savez_compressed(file,**s)
        summary.append(dict(kind=kind,config=conf.name,**summarize(s,p,a,common_value=value)))
        daily.extend(dict(kind=kind,config=conf.name,date=str(dates[d]),cash=float(s['cash'][d].sum())) for d in range(31,365))
        write_csv(OUT/f'experiments/q{kind}_candidate_summary.csv',summary)
        print('Q',kind,'diagnostic',conf.name,flush=True)
    write_csv(OUT/f'experiments/q{kind}_candidate_daily.csv',daily)
    variations={
      'oracle_P':dict(oracle='P'),'oracle_NP':dict(oracle='NP'),
      'stale_price':dict(stale_price=True),'decision_lock':dict(locked=True),
      'only0':dict(updates=()),'current_lag':dict(current_known=False)}
    if kind==43:variations.update(stale_pv=dict(stale_pv=True),settlement_B=dict(rule='B'),settlement_B_net=dict(rule='B_net'))
    else:variations['official_pv0']=dict(pv0=True)
    comparisons={}
    for name,kwargs in variations.items():
        file=OUT/f'experiments/q{kind}_{name}.npz'
        if file.exists():s=dict(np.load(file))
        else:s,_=replay(f,kind,calendar=cal,**kwargs);np.savez_compressed(file,**s)
        comparisons[name]=summarize(s,p,a,common_value=value)
        print('Q',kind,name,comparisons[name]['total_cost_yuan'],flush=True)
    # Exact matched-family experiments at preregistered mean7 q=.7 baseline.
    for name,conf,kwargs in [('paired_E0',Config(),{}),('paired_E1',Config(execution=36),{}),
      ('paired_joint',Config(risk='joint'),{}),('independent_price',Config(risk='joint'),dict(independent=True)),
      ('paired_CVaR',Config(risk='joint',beta=.1),{})]:
        s,_=replay(f,kind,conf=conf,**kwargs);np.savez_compressed(OUT/f'experiments/q{kind}_{name}.npz',**s)
        comparisons[name]=summarize(s,p,a,common_value=value)
    # Frozen trajectories are read, never reoptimized or overwritten.
    old=dict(np.load(ROOT/f'Q{kind-40}/experiments/formal_arrays.npz'))
    g0=old['G'] if kind==42 else old['versions'][:,0]
    old['plan_cost']=g0*p;old['emergency_cost']=5*p*old['H']
    if kind==42:
        old['versions']=np.repeat(old['G'][:,None],4,axis=1);old['U']=np.zeros((365,3,144));old['R']=old['U'].copy()
    old['up_cost']=1.5*p*old['U'].sum(1);old['penalty']=.5*p*old['R'].sum(1);old['refund']=np.zeros_like(p)
    old['cash']=old['plan_cost']+old['up_cost']+old['penalty']+old['emergency_cost']
    comparisons['frozen_repriced']=summarize(old,p,a,common_value=value)
    np.savez_compressed(OUT/f'experiments/q{kind}_frozen_repriced.npz',**old)
    # Full clairvoyant ordinary purchase is a relaxation: same formal initial AND final SOC.
    formal=dict(np.load(OUT/f'experiments/q{kind}_formal.npz'))
    from microgrid_online import plan_net
    oracle=plan_net((a[31:,:,0]-a[31:,:,1]).ravel(),p[31:].ravel(),float(formal['E'][31,0]),float(formal['E'][-1,-1]))
    comparisons['full_oracle_same_boundary']=dict(total_cost_yuan=float(p[31:].ravel()@oracle['G']),initial_kwh=float(oracle['E'][0]),terminal_kwh=float(oracle['E'][-1]))
    np.savez_compressed(OUT/f'experiments/q{kind}_full_oracle.npz',**oracle)
    write_json(OUT/f'experiments/q{kind}_comparisons.json',comparisons)
    check_protected()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--compute-kind',type=int,choices=[42,43]);parser.add_argument('--diagnostics-kind',type=int,choices=[42,43]);parser.add_argument('--deliver',action='store_true');args=parser.parse_args()
    if args.prepare:
        from prepare_q4 import prepare;prepare()
    elif args.precheck:
        from price_diagnostics import price_diagnostics;price_diagnostics()
        r=subprocess.run([sys.executable,str(OUT/'test_q4.py')],capture_output=True,text=True)
        (OUT/'core_tests.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('Core tests failed; see Q4/core_tests.log')
        write_json(OUT/'experiments/core_tests_passed.json',dict(exit_code=r.returncode))
    elif args.compute_kind:compute_kind(args.compute_kind)
    elif args.diagnostics_kind:diagnostics_kind(args.diagnostics_kind)
    elif args.deliver:
        from report_q4 import deliver;deliver()
    else:
        for flag in ('--prepare','--precheck'):subprocess.run([sys.executable,__file__,flag],check=True)
        for flag in ('--compute-kind','--diagnostics-kind'):
            processes=[]
            for kind in (42,43):
                log=(OUT/f'q{kind}_{flag[2:]}.log').open('w');processes.append((subprocess.Popen([sys.executable,__file__,flag,str(kind)],stdout=log,stderr=subprocess.STDOUT),log))
            for proc,log in processes:
                code=proc.wait();log.close()
                if code:raise RuntimeError(f'{flag} failed')
        subprocess.run([sys.executable,__file__,'--deliver'],check=True)
        subprocess.run([sys.executable,str(OUT/'audit_q4.py')],check=True)

if __name__=='__main__':main()
