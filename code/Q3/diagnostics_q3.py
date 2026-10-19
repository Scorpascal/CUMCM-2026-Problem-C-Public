"""补充公平性诊断：仅0时独立历史选参；同权限逐阶段屏蔽新PV；18h次日信息。

在基础消融结果后添加的诊断，不改变正式选择和原预注册候选。
"""
from dataclasses import replace
from datetime import datetime,timedelta
import numpy as np
from solve_q3 import OUT,inputs,write_csv,write_json,replay,select_config
from q3_forecast import FeatureFactory
from q3_model import Config,CANDIDATES,run_day,stack_days,summarize

def supplemental():
    dates,a,p,records=inputs();f=FeatureFactory(a,records)
    formal=dict(np.load(OUT/'experiments/formal_arrays.npz'))
    warmup=[{k:v[d] for k,v in formal.items()} for d in range(31)]
    calendar=[Config(**v) for v in __import__('json').loads((OUT/'experiments/calendar.json').read_text())]
    write_json(OUT/'experiments/supplement_protocol.json',dict(added_after_initial_ablations=True,reason='separate adjustment access, new PV information, and retuning of nonadjustable baseline',changes_to_formal_policy=False,zero_only_selection='same 16 candidates, past 21 days monthly',freshness='same calendar and all adjustment rights; withhold one new release',next_day18='preserve today forecast; replace next-day18 release targets with prior available prediction'))
    rows=[]
    for k in (1,2,3):
        ss,_=replay(f,p,warmup,calendar,stale=(k,))
        np.savez_compressed(OUT/f'experiments/withhold_new_{6*k}.npz',**ss)
        rows.append(dict(policy=f'withhold_new_{6*k}_same_rights',**summarize(ss,p)))
    # 独立仅0时基准：不是将可调整低q日历机械用于不可调整方案。
    known=[float(formal['E'][0,0])]+[float(formal['E'][d,-1]) for d in range(31)];e=known[-1]
    days=list(warmup);selection=[];conf=Config()
    for d in range(31,365):
        if dates[d].day==1:
            conf,rr=select_config(f,d,known,p,updates=());selection.extend(rr)
            print('0h-only monthly selection',dates[d],conf.name,flush=True)
        ss,_=run_day(f,d,e,p,conf,updates=());days.append(ss);e=float(ss['E'][-1]);known.append(e)
    ss=stack_days(days);np.savez_compressed(OUT/'experiments/zero_only_selected.npz',**ss)
    write_csv(OUT/'experiments/zero_only_monthly_selection.csv',selection)
    rows.append(dict(policy='zero_only_independent_past_selection',**summarize(ss,p)))
    # 18时只保留当日目标，次日目标用12时之前已知曲线补齐，不能使用次日实际。
    changed=dict(records)
    for d in range(365):
        issue=datetime(2025,1,1)+timedelta(days=d,hours=18);rec=records[issue]
        old=f.raw(d,3,'linear',0.,2);powers=list(rec.power)
        for h in range(7,25):powers[h-1]=float(old[h*6-1,1]*6)
        changed[issue]=replace(rec,power=tuple(powers))
    ff=FeatureFactory(a,changed)
    ss,_=replay(ff,p,warmup,Config(horizon='R48'))
    rows.append(dict(policy='R48_18_nextday_old_information',**summarize(ss,p)))
    np.savez_compressed(OUT/'experiments/R48_18_nextday_old.npz',**ss)
    write_csv(OUT/'experiments/supplemental.csv',rows)
    print('supplemental causal diagnostics complete',flush=True)

if __name__=='__main__':supplemental()
