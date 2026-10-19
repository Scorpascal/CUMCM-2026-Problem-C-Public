"""问题2唯一重建入口：原始附件→历史选择→全年执行→实验→交付→测试→冻结。

python Q2/solve_q2.py --compute 仅计算；--deliver 用本次计算产物生成图表/论文；
无参数完整重建（已有冻结时拒绝覆盖）。所有派生文件生成代码均保留。
"""
from pathlib import Path
import os,sys,time,json,csv,hashlib,argparse
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
import numpy as np
from microgrid_common import Battery
from microgrid_online import read_year,execute_day,plan_net,terminal_value,validate_execution,merge_emergency
from forecast_q2 import build_features,METHODS
from strategy_q2 import Config,CANDIDATES,WARMUP,make_plan,select_config


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')


def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def check_q1():
    handoff=OUT/'q1_handoff_hashes.json'
    if not handoff.exists():
        return 0  # Submission copy omits freeze-only protection metadata.
    known=json.loads(handoff.read_text())
    changes=[p for p,h in known.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changes, f'Q1接手后发生变化: {changes}'
    return len(known)


def blank_result(n):
    result={k:np.zeros((n,144)) for k in ('G','C','D','H','W','unused_grid','pv_curtail','pv_curtail_upper','C_plan','D_plan','margin')}
    result['E']=np.zeros((n,145));result['E_plan']=np.zeros((n,145))
    result['prediction']=np.zeros((n,144,2));result['q']=np.zeros(n);result['seconds']=np.zeros(n)
    result['config']=np.full(n,'',dtype='<U80')
    return result


def store_day(out,day,real,plan):
    for k in ('G','C','D','H','W','E','unused_grid','pv_curtail','pv_curtail_upper'):
        out[k][day]=real[k]
    if plan is not None:
        for k in ('C','D','E'):out[k+'_plan'][day]=plan[k][:145 if k=='E' else 144]
        for k in ('prediction','margin','q','seconds','config'):out[k][day]=plan[k]
    else:
        out['E_plan'][day]=real['E'];out['config'][day]='startup_idle_zero_plan'


def simulate(dates,actual,price,features,config=None,battery=Battery(),fixed_actions=False,
             no_storage=False,stop=None,candidates=CANDIDATES,write_progress=False):
    """环境读取actual，日前make_plan仅能读取因果features和当前E。"""
    n=len(actual) if stop is None else stop
    out=blank_result(n);e=6000.;known=[e];selection=[];current=WARMUP
    for day in range(n):
        if day==0:
            plan=None;g=np.zeros(144)
        else:
            if day<31:current=WARMUP
            elif config is not None:current=config
            elif dates[day].day==1:
                current,rows=select_config(features[:day],actual[:day],known,day,price,battery,candidates)
                selection.extend(rows)
                if write_progress:
                    print(f'月首选择 {dates[day]} {current.name}',flush=True)
                    write_csv(OUT/'experiments/monthly_selection_progress.csv',selection)
            plan=make_plan(features[day],e,price,current,battery)
            g=plan['G'][:144].copy()
            if no_storage and day>=31:
                g=np.maximum(plan['prediction'][:,0]-plan['prediction'][:,1],0)
                plan['C'][:144]=0;plan['D'][:144]=0;plan['E'][:145]=e
                plan['config']='no_storage_'+current.method
        fixed=(plan['C'][:144],plan['D'][:144]) if fixed_actions and day>=31 else None
        real=execute_day(g,actual[day],e,battery,fixed=fixed,idle=day==0 or (no_storage and day>=31))
        store_day(out,day,real,plan);e=float(real['E'][-1]);known.append(e)
    return out,selection


def summary_of(s,actual,price,battery=Battery(),start=31):
    g,h,e=s['G'][start:],s['H'][start:],s['E'][start:]
    planned=g@price;emergency=h@(5*price);daily=planned+emergency
    errors=[]
    for d in range(start,len(s['G'])):
        errors.append(validate_execution(actual[d],{k:s[k][d] for k in ('G','C','D','H','W','E')},battery))
    result=dict(days=len(g),plan_cost_yuan=float(planned.sum()),emergency_cost_yuan=float(emergency.sum()),
        total_cost_yuan=float(daily.sum()),daily_mean_yuan=float(daily.mean()),daily_p95_yuan=float(np.quantile(daily,.95)),
        plan_kwh=float(g.sum()),emergency_kwh=float(h.sum()),total_grid_kwh=float(g.sum()+h.sum()),
        emergency_periods=int((h>1e-7).sum()),emergency_days=int(np.any(h>1e-7,axis=1).sum()),
        max_emergency_kwh=float(h.max()),energy_min=float(e.min()),energy_max=float(e.max()),
        initial_kwh=float(e[0,0]),terminal_kwh=float(e[-1,-1]),
        charge_kwh=float(s['C'][start:].sum()),discharge_kwh=float(s['D'][start:].sum()),
        waste_kwh=float(s['W'][start:].sum()),unused_plan_kwh=float(s['unused_grid'][start:].sum()),
        pv_curtail_kwh=float(s['pv_curtail'][start:].sum()),pv_curtail_upper_kwh=float(s['pv_curtail_upper'][start:].sum()),
        pv_utilization=float(1-s['pv_curtail'][start:].sum()/actual[start:len(s['G']),:,1].sum()),
        continuity_error=float(np.max(np.abs(e[1:,0]-e[:-1,-1]))) if len(e)>1 else 0.,
        lower_bound_states=int((e<=battery.minimum+1e-6).sum()),upper_bound_states=int((e>=battery.maximum-1e-6).sum()),
        avg_plan_solve_seconds=float(s['seconds'][start:].mean()),
        terminal_adjusted_cost_yuan=float(daily.sum()-terminal_value(price,battery)*(e[-1,-1]-e[0,0])),
        checks={k:max(r[k] for r in errors) for k in errors[0]})
    return result


def forecasting_metrics(dates,actual,features,formal):
    rows=[]
    for method in (*METHODS,'formal'):
        pred=np.asarray([f.prediction[method][0] for f in features]) if method!='formal' else formal['prediction']
        for group in ['all']+[f'month_{m:02d}' for m in range(2,13)]+[f'hour_{h:02d}' for h in range(24)]:
            ds=np.arange(31,len(dates));ts=np.arange(144)
            if group.startswith('month'):ds=np.array([d for d in ds if dates[d].month==int(group[-2:])])
            if group.startswith('hour'):ts=np.arange(int(group[-2:])*6,(int(group[-2:])+1)*6)
            for obj,j in [('load',0),('pv',1),('net',2)]:
                v=actual[:,:,j]-pred[:,:,j] if j<2 else (actual[:,:,0]-actual[:,:,1])-(pred[:,:,0]-pred[:,:,1])
                err=v[np.ix_(ds,ts)]*6
                row=dict(method=method,group=group,object=obj,mae_kw=float(np.abs(err).mean()),rmse_kw=float(np.sqrt((err**2).mean())),bias_kw=float(err.mean()),coverage='')
                if j==2 and method=='formal':row['coverage']=float((v[np.ix_(ds,ts)]<=formal['margin'][np.ix_(ds,ts)]+1e-7).mean())
                rows.append(row)
    return rows


def fixed_path_sensitivity(dates,actual,price,features,formal,configs,battery=Battery(),fixed=False):
    """冻结已选日历配置重放；用于口径影响，非重新挑参后的最优值。"""
    out=blank_result(len(actual));e=6000.
    for d in range(len(actual)):
        if d==0:plan=None;g=np.zeros(144)
        else:
            conf=WARMUP if d<31 else configs[formal['config'][d]]
            plan=make_plan(features[d],e,price,conf,battery);g=plan['G'][:144]
        real=execute_day(g,actual[d],e,battery,
                         fixed=(plan['C'][:144],plan['D'][:144]) if fixed and d>=31 else None,idle=d==0)
        store_day(out,d,real,plan);e=real['E'][-1]
    return out


def pressure_tests(dates,actual,price,features,formal):
    """历史经验整日场景对照独立重排；不声称独立真实分布置信保证。"""
    rng=np.random.default_rng(20260911);rows=[]
    selected=[i for i,d in enumerate(dates) if d.isoformat() in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21')]
    for day in selected:
        conf=next(c for c in CANDIDATES if c.name==formal['config'][day])
        r=features[day].residual[conf.method][0];pred=formal['prediction'][day]
        for kind in ('whole_day','independent_points'):
            sims=[];hs=[];eds=[]
            for k in range(200):
                error=r[rng.integers(len(r))] if kind=='whole_day' else r[rng.integers(len(r),size=144),np.arange(144)]
                scene=np.maximum(pred+error,0)
                real=execute_day(formal['G'][day],scene,formal['E'][day,0])
                sims.append(float(price@real['G']+5*price@real['H']));hs.append(real['H'].sum());eds.append(real['E'][-1])
            rows.append(dict(date=dates[day].isoformat(),kind=kind,samples=200,source_max_day=max(features[day].residual_dates[conf.method][0]),
                       mean_cost=float(np.mean(sims)),p95_cost=float(np.quantile(sims,.95)),mean_emergency=float(np.mean(hs)),mean_terminal=float(np.mean(eds))))
    return rows


def compute():
    check_q1();tic=time.perf_counter()
    dates,actual,price=read_year();features=build_features(actual)
    configs={c.name:c for c in CANDIDATES}
    write_json(OUT/'experiments/protocol.json',dict(candidates=[asdict(c) for c in CANDIDATES],
        validation_days=21,residual_days=30,selection='monthly_past_only',warmup=asdict(WARMUP),
        pricing='p*G + 5*p*H',formal_start='2025-02-01',seed=20260911))
    formal,selection=simulate(dates,actual,price,features,write_progress=True)
    np.savez_compressed(OUT/'experiments/formal_arrays.npz',**formal)
    write_csv(OUT/'experiments/monthly_selection.csv',selection)
    fm=summary_of(formal,actual,price)
    write_json(OUT/'experiments/formal_metrics.json',fm)
    write_csv(OUT/'forecast_metrics.csv',forecasting_metrics(dates,actual,features,formal))
    print('正式策略现金费用',fm['total_cost_yuan'],flush=True)
    comparisons=[];candidate_daily=[]
    for i,conf in enumerate(CANDIDATES):
        s,_=simulate(dates,actual,price,features,config=conf)
        m=summary_of(s,actual,price)
        comparisons.append(dict(config=conf.name,method=conf.method,q=conf.q,horizon_hours=24*conf.horizon,
                                terminal=conf.terminal,scenario=conf.scenario,**{k:v for k,v in m.items() if k!='checks'}))
        for d in range(31,365):
            candidate_daily.append(dict(config=conf.name,date=dates[d].isoformat(),
                plan_cost=float(price@s['G'][d]),emergency_cost=float(5*price@s['H'][d]),
                emergency_kwh=float(s['H'][d].sum()),initial=float(s['E'][d,0]),terminal=float(s['E'][d,-1])))
        if i%8==0:print(f'固定候选诊断 {i+1}/{len(CANDIDATES)} {conf.name} {m["total_cost_yuan"]:.2f}',flush=True)
    write_csv(OUT/'experiments/candidate_summary.csv',comparisons)
    write_csv(OUT/'experiments/candidate_daily.csv',candidate_daily)
    s,_=simulate(dates,actual,price,features,config=Config('mean7',0,1),no_storage=True)
    sensitivities={'no_storage_mean7':summary_of(s,actual,price)}
    for name,b,fixed in [('fixed_battery',Battery(),True),('roundtrip_90',replace(Battery(),eta_charge=np.sqrt(.9),eta_discharge=np.sqrt(.9)),False)]:
        s=fixed_path_sensitivity(dates,actual,price,features,formal,configs,b,fixed)
        sensitivities[name]=summary_of(s,actual,price,b)
    write_json(OUT/'experiments/sensitivities.json',sensitivities)
    print('计算整段同边界完美信息下界',flush=True)
    oracle=plan_net((actual[31:,:,0]-actual[31:,:,1]).ravel(),np.tile(price,334),
                    formal['E'][31,0],terminal=float(formal['E'][-1,-1]))
    np.savez_compressed(OUT/'experiments/oracle_arrays.npz',**{k:oracle[k] for k in ('G','C','D','E','W')})
    write_json(OUT/'experiments/oracle.json',{k:v for k,v in oracle.items() if k not in ('G','C','D','E','W')})
    assert oracle['objective']<=fm['total_cost_yuan']+1e-5
    write_csv(OUT/'experiments/scenario_pressure.csv',pressure_tests(dates,actual,price,features,formal))
    audit=[]
    for d in range(365):
        f=features[d]; audit.append(dict(date=dates[d].isoformat(),training_max=(dates[d-1].isoformat() if d else ''),
                    history_days=d,history_sha256=f.history_sha256,config=str(formal['config'][d]),q=float(formal['q'][d]),
                    max_residual_observed_index=max([-1]+[x for m in METHODS for js in f.residual_dates[m] for x in js])))
    write_csv(OUT/'information_audit.csv',audit)
    write_json(OUT/'experiments/run_metadata.json',dict(total_seconds=time.perf_counter()-tic,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'附件/附件1.xlsx',ROOT/'附件/附件2.xlsx',ROOT/'附件/附件5/result2.xlsx',ROOT/'C题.pdf'] if p.exists()},
        q1_unchanged_files=check_q1(),gate_a_b='written_before_solver',gate_c='prototype.json',
        official_method='monthly past-only selection among predeclared causal policies',
        fixed_candidate_ranking='retrospective diagnostics only; NOT used for official past plans'))
    print('全部计算完成',time.perf_counter()-tic,flush=True)


def prototype():
    """重建Gate C小规模证明；全年正式结果不参与本检查。"""
    dates,a,p=read_year();features=build_features(a[:35]);e=6000.;records=[]
    for day in range(1,5):
        s=make_plan(features[day],e,p,Config())
        r=execute_day(s['G'][:144],a[day],e)
        m=plan_net(s['net'],np.tile(p,2),e,integer=True)
        assert abs(s['objective']-m['objective'])<1e-5
        records.append(dict(day=day,lp_milp_difference=s['objective']-m['objective'],
                            checks=validate_execution(a[day],r),seconds=s['seconds']))
        e=r['E'][-1]
    changed=a[:35].copy();changed[20:]*=2
    alt=build_features(changed)
    x=make_plan(features[20],6000,p,Config());y=make_plan(alt[20],6000,p,Config())
    np.testing.assert_array_equal(x['G'],y['G'])
    write_json(OUT/'experiments/prototype.json',dict(days=records,counterfactual_plan_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute',action='store_true');parser.add_argument('--deliver',action='store_true')
    parser.add_argument('--prototype',action='store_true')
    args=parser.parse_args()
    if (OUT/'freeze_manifest.json').exists():raise RuntimeError('Q2已冻结；请在独立副本解除清单后复现')
    if args.prototype:
        prototype();sys.exit(0)
    if not args.deliver:compute()
    if not args.compute:
        from report_q2 import deliver
        deliver()
