"""原始附件→预审→原型→因果月度选择→正式全年→消融→图表/论文→验收。
所有问题3派生文件由此入口和audit_q3.py/freeze_q3.py重建。
"""
import sys,os,json,csv,time,argparse,hashlib
from pathlib import Path
from datetime import datetime,timedelta
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
import numpy as np
from microgrid_online import read_year,terminal_value
from microgrid_forecasts import read_pv_forecasts
from q3_forecast import FeatureFactory
from q3_model import Config,CANDIDATES,WARMUP,run_day,stack_days,summarize

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def write_csv(path,rows,fields=None):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]) if rows else fields or ['empty']);w.writeheader();w.writerows(rows)

def check_protected():
    handoff=OUT/'handoff_hashes.json'
    if not handoff.exists():
        return 0  # Submission copy omits freeze-only protection metadata.
    known=json.loads(handoff.read_text())
    changed=[p for p,h in known.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    return len(known)

def inputs():
    dates,actual,price=read_year();records=read_pv_forecasts(ROOT/'附件/附件3.xlsx')
    return dates,actual,price,records

def prototype():
    from prepare_q3 import prepare
    prepare();check_protected();dates,a,p,r=inputs();factory=FeatureFactory(a,r);rows=[]
    tic=time.perf_counter()
    for begin in (28,77,169,351):
        for conf in CANDIDATES:
            e=6000.;days=[]
            for d in range(begin,begin+4):
                s,events=run_day(factory,d,e,p,conf,details=True);days.append(s);e=s['E'][-1]
            agg=summarize(stack_days(days),p,start=0)
            rows.append(dict(begin=dates[begin].isoformat(),config=conf.name,**agg))
        print('prototype continuous window',dates[begin],flush=True)
    write_csv(OUT/'experiments/prototype.csv',rows)
    write_json(OUT/'experiments/prototype.json',dict(windows=4,days_each=4,candidates=len(CANDIDATES),physics=True,seconds=time.perf_counter()-tic))

def select_config(factory,day,known,price,candidates=CANDIDATES,updates=(1,2,3)):
    # 选参接口显式截断真实数据和预报档案，验证日每个阶段再按发行时间隔离。
    cutoff=datetime(2025,1,1)+timedelta(days=day)
    past=FeatureFactory(factory.actual[:day],{t:v for t,v in factory.records.items() if t<cutoff})
    assert len(past.actual)==day and all(t<cutoff for t in past.records)
    begin=max(1,day-21);start=float(known[begin]);rows=[]
    for conf in candidates:
        e=start;cash=0.;h=0.
        for d in range(begin,day):
            s,_=run_day(past,d,e,price,conf,updates=updates);cash+=s['cash'].sum();h+=s['H'].sum();e=s['E'][-1]
        score=cash-terminal_value(price)*(e-start)
        rows.append(dict(selection_day=day,begin=begin,end=day-1,config=conf.name,cash=float(cash),score=float(score),initial=start,terminal=float(e),emergency_kwh=float(h)))
    idx=min(range(len(rows)),key=lambda i:(rows[i]['score'],i))
    for i,row in enumerate(rows):row['selected']=i==idx
    # lru_cache是实例方法全局缓存，清理避免保存此前factory历史副本。
    past.feature.cache_clear();past.raw.cache_clear()
    return candidates[idx],rows

def replay(factory,p,warmup,calendar,updates=(1,2,3),rule='A',stale=False,details=False):
    days=list(warmup);events=[];e=float(days[-1]['E'][-1])
    for d in range(31,len(factory.actual)):
        conf=calendar[d] if isinstance(calendar,list) else calendar
        s,ev=run_day(factory,d,e,p,conf,updates,rule,stale,details=details)
        days.append(s);events.extend(ev);e=s['E'][-1]
    return stack_days(days),events

def compute():
    check_protected();assert (OUT/'experiments/prototype.json').exists()
    import subprocess
    core=subprocess.run([sys.executable,str(OUT/'test_q3.py'),'--core'],capture_output=True,text=True)
    (OUT/'experiments/core_tests.log').write_text(core.stdout+core.stderr)
    assert core.returncode==0,'核心测试未通过，拒绝正式全年运行'
    dates,a,p,r=inputs();f=FeatureFactory(a,r);tic=time.perf_counter()
    days=[];warmup=[];known=[6000.];e=6000.;selection=[];events=[];calendar=[];conf=WARMUP
    for d in range(365):
        if d>=31 and dates[d].day==1:
            conf,rows=select_config(f,d,known,p);selection.extend(rows)
            print('monthly selection',dates[d],conf.name,flush=True)
            write_csv(OUT/'experiments/monthly_selection.csv',selection)
        s,ev=run_day(f,d,e,p,conf,details=True);days.append(s);events.extend(ev)
        if d<31:warmup.append(s)
        calendar.append(conf);e=float(s['E'][-1]);known.append(e)
    formal=stack_days(days);np.savez_compressed(OUT/'experiments/formal_arrays.npz',**formal)
    write_json(OUT/'experiments/calendar.json',[asdict(c) for c in calendar])
    write_csv(OUT/'voi_events.csv',[dict(date=dates[v['day']].isoformat(),decision_time=f"{v['stage']*6}:00",**v) for v in events if v['day']>=31])
    write_json(OUT/'summary.json',summarize(formal,p))
    write_json(OUT/'experiments/run_metadata.json',dict(formal_completed_before_fixed_rankings=True,formal_seconds=time.perf_counter()-tic))
    print('formal complete',summarize(formal,p)['total_cost_yuan'],flush=True)
    # 一旦正式轨迹完成才查看固定策略的全年事后排名。
    candidates=[];daily=[]
    for conf in CANDIDATES:
        ss,_=replay(f,p,warmup,conf)
        candidates.append(dict(config=conf.name,**summarize(ss,p)))
        daily.extend(dict(date=dates[d].isoformat(),config=conf.name,cash=float(ss['cash'][d].sum()),initial=float(ss['E'][d,0]),terminal=float(ss['E'][d,-1]),emergency=float(ss['H'][d].sum())) for d in range(31,365))
        write_csv(OUT/'experiments/candidate_summary.csv',candidates)
        print('fixed diagnostic',conf.name,candidates[-1]['total_cost_yuan'],flush=True)
    write_csv(OUT/'experiments/candidate_daily.csv',daily)
    results=[]
    for mask in range(8):
        subset=tuple(k for k in (1,2,3) if mask & (1<<(k-1)))
        ss,_=replay(f,p,warmup,calendar,subset)
        name='0'+''.join('_'+str(k*6) for k in subset)
        np.savez_compressed(OUT/f'experiments/subset_{mask}.npz',**ss)
        results.append(dict(policy=name,mask=mask,**summarize(ss,p)))
        print('information subset',name,results[-1]['total_cost_yuan'],flush=True)
    ss,_=replay(f,p,warmup,calendar,stale=True)
    np.savez_compressed(OUT/'experiments/stale_arrays.npz',**ss)
    results.append(dict(policy='old_forecast_same_adjustment_rights',mask=-1,**summarize(ss,p)))
    write_csv(OUT/'experiments/information_ablation.csv',results)
    sensitivities=[]
    for rule in ('B','B_net'):
        ss,_=replay(f,p,warmup,calendar,rule=rule)
        np.savez_compressed(OUT/f'experiments/settlement_{rule}.npz',**ss)
        sensitivities.append(dict(policy='settlement_'+rule,**summarize(ss,p)))
        print('settlement sensitivity',rule,sensitivities[-1]['total_cost_yuan'],flush=True)
    for horizon in ('T24','R48'):
        conf=Config(horizon=horizon)
        for subset in ((1,2),(1,2,3)):
            ss,_=replay(f,p,warmup,conf,updates=subset)
            sensitivities.append(dict(policy=horizon+('_with18' if 3 in subset else '_without18'),**summarize(ss,p)))
    write_csv(OUT/'experiments/sensitivities.csv',sensitivities)
    from diagnostics_q3 import supplemental
    supplemental()
    write_json(OUT/'experiments/run_metadata.json',dict(formal_completed_before_fixed_rankings=True,total_seconds=time.perf_counter()-tic,protected_files=check_protected(),configuration='monthly_past_only',posthoc_not_used_for_selection=True))
    print('all experiments complete',time.perf_counter()-tic,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prototype',action='store_true');parser.add_argument('--compute',action='store_true');parser.add_argument('--deliver',action='store_true');args=parser.parse_args()
    if (OUT/'freeze_manifest.json').exists():raise RuntimeError('Q3已冻结，请在独立复现副本移走清单后运行')
    if args.prototype:prototype()
    elif args.compute:compute()
    elif args.deliver:
        from report_q3 import deliver
        deliver()
    else:
        prototype()
        import subprocess
        subprocess.run([sys.executable,str(OUT/'test_q3.py'),'--core'],check=True)
        compute()
        from report_q3 import deliver
        deliver()
