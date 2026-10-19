"""独立验收：从CSV底层和原始输入重新聚合，不用summary作为计算源。"""
import csv,json,sys,subprocess,time,hashlib
from pathlib import Path
from decimal import Decimal
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from microgrid_online import read_year,PERIODS,merge_emergency
from solve_q3 import write_json,check_protected

def rows(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def audit(run_tests=True):
    tic=time.perf_counter();dates,a,p=read_year();data=rows(OUT/'dispatch_10min.csv');ledger=rows(OUT/'adjustment_ledger.csv')
    assert len(data)==334*144 and len(ledger)==334*216
    def arr(k):return np.array([float(r[k]) for r in data]).reshape(334,144)
    for d in range(334):
        part=data[d*144:(d+1)*144]
        assert [r['date'] for r in part]==[str(dates[d+31])]*144
        assert [r['period'] for r in part]==PERIODS and [int(r['index']) for r in part]==list(range(144))
    np.testing.assert_allclose(arr('load_kw')/6,a[31:,:,0],rtol=0,atol=1e-9)
    np.testing.assert_allclose(arr('pv_kw')/6,a[31:,:,1],rtol=0,atol=1e-9)
    np.testing.assert_allclose(arr('price_yuan_kwh'),np.tile(p,(334,1)),rtol=0,atol=1e-12)
    g=arr('G_effective');c=arr('C_real');d=arr('D_real');h=arr('H_real');w=arr('W_real');e=arr('E_start');ee=arr('E_end')
    balance=g+a[31:,:,1]+d+h-a[31:,:,0]-c-w
    state=ee-e-.9*c+d/.9
    np.testing.assert_allclose(balance,0,atol=1e-7);np.testing.assert_allclose(state,0,atol=1e-7)
    np.testing.assert_array_equal(ee.ravel()[:-1],e.ravel()[1:])
    assert min(e.min(),ee.min())>=1200-1e-7 and max(e.max(),ee.max())<=10800+1e-7
    assert max(c.max(),d.max())<=5000/6+1e-7 and not np.any((c>1e-7)&(d>1e-7))
    short=a[31:,:,0]-a[31:,:,1]-g
    dhat=np.minimum(np.minimum(np.maximum(short,0),5000/6),np.maximum(0,(e-1200)*.9))
    np.testing.assert_allclose(h,np.maximum(0,short-dhat),atol=1e-7)
    # 财务从最底层p、G0、每笔U/R、H重算，Decimal避免累积浮点解释混淆。
    D=Decimal;zero=D(0)
    plan=sum((D(r['price_yuan_kwh'])*D(r['G0']) for r in data),zero)
    emergency=sum((D(5)*D(r['price_yuan_kwh'])*D(r['H_real']) for r in data),zero)
    up=sum((D('1.5')*D(r['normal_price'])*D(r['increase_U']) for r in ledger),zero)
    down=sum((D('.5')*D(r['normal_price'])*D(r['decrease_R']) for r in ledger),zero)
    increase=sum((D(r['increase_U']) for r in ledger),zero);decrease=sum((D(r['decrease_R']) for r in ledger),zero)
    hp=sum((D(r['H_real']) for r in data),zero);total=plan+up+down+emergency
    values=dict(plan_cost_yuan=plan,up_cost_yuan=up,penalty_yuan=down,refund_yuan=zero,emergency_cost_yuan=emergency,total_cost_yuan=total,
      increase_kwh=increase,decrease_kwh=decrease,emergency_kwh=hp,emergency_periods=int(np.sum(h>1e-7)))
    summary=json.loads((OUT/'summary.json').read_text())
    for k,v in values.items():assert abs(float(v)-summary[k])<1e-6,(k,v,summary[k])
    # 不使用NPZ，独立按账本逐笔恢复承诺，最终与CSV有效G逐点核对。
    commitment={(r['date'],int(r['index'])):D(r['G0']) for r in data}
    seen=set();max_ledger_error=0
    for row in ledger:
        key=(row['delivery_date'],int(row['index']));k=int(row['stage']);stamp=(key,k)
        assert stamp not in seen;seen.add(stamp)
        assert int(row['index'])>=k*36 and row['decision_date']==row['delivery_date']
        old=D(row['previous_commitment']);new=D(row['new_commitment']);u=D(row['increase_U']);r=D(row['decrease_R'])
        assert u>=0 and r>=0 and not(u>D('1e-7') and r>D('1e-7'))
        err=max(abs(commitment[key]-old),abs(new-old-u+r));max_ledger_error=max(max_ledger_error,float(err));assert err<D('1e-6')
        commitment[key]=new
        assert abs(D(row['adjustment_charge'])-D(row['normal_price'])*(D('1.5')*u+D('.5')*r))<D('1e-6')
    for row in data:assert abs(commitment[(row['date'],int(row['index']))]-D(row['G_effective']))<D('1e-6')
    merged=merge_emergency(dates[31:],h);saved=rows(OUT/'emergency_purchase_merged.csv');assert len(merged)==len(saved)
    for x,y in zip(merged,saved):
        assert x['date']==y['date'] and x['period']==y['period'];assert abs(x['kwh']-float(y['kwh']))<1e-6
    # 正式配置日历内，每个选参训练区间必须严格在月首前；核验不据摘要声称通过。
    chosen=rows(OUT/'experiments/monthly_selection.csv')
    for row in chosen:assert int(row['end'])<int(row['selection_day'])
    check_protected();tests={}
    if run_tests:
        for name,cmd in [('q1_tests',[sys.executable,str(ROOT/'Q1/test_q1.py')]),('q2_tests',[sys.executable,str(ROOT/'Q2/test_q2.py')]),('q3_tests',[sys.executable,str(OUT/'test_q3.py')]),
                         ('q1_freeze',[sys.executable,str(ROOT/'Q1/freeze_q1.py')]),('q2_freeze',[sys.executable,str(ROOT/'Q2/freeze_q2.py')])]:
            if name.endswith('_freeze') and not Path(cmd[1]).exists():
                tests[name]='not_packaged';continue
            result=subprocess.run(cmd,capture_output=True,text=True)
            (OUT/(name+'.log')).write_text(result.stdout+result.stderr);tests[name]=result.returncode
            if result.returncode:raise RuntimeError(name+' failed; see log')
    protected=check_protected()
    result=dict(status='PASS',independent_source='original inputs + dispatch_10min.csv + adjustment_ledger.csv; summary used only after recomputation',
      decimal_aggregates={k:str(v) for k,v in values.items()},formal_days=334,intervals=len(data),ledger_rows=len(ledger),
      balance_max_kwh=float(np.abs(balance).max()),state_max_kwh=float(np.abs(state).max()),ledger_max_kwh=max_ledger_error,
      excel_full_readback='q3_tests.log OutputTests.test_excel_all_values' if run_tests else 'not_run',regression_exit_codes=tests,protected_files=protected,seconds=time.perf_counter()-tic,
      warnings=['A/B/B_net settlement ambiguity','10min endpoint and efficiency each90% assumptions','instantaneous feedback aggregate','different initial/end SOC compared with Q2','shared-calendar vs independently selected ablations','local realized VOI is not additive annual contribution'])
    write_json(OUT/'validation.json',result)
    lines=['# Q3最终验收审计','','结论：PASS。独立读取原始输入、48,096条交付时段和72,144条调整账本，用十进制数重算下表；summary仅在聚合后用于交叉对照，不作为计算源。',
      '','|项目|底层重新计算|','|---|---|']
    lines.extend(f'|{k}|{v}|' for k,v in values.items())
    lines+=['',f'逐时平衡最大残差{result["balance_max_kwh"]:.3e} kWh，状态递推最大残差{result["state_max_kwh"]:.3e} kWh；逐笔承诺恢复最大误差{max_ledger_error:.3e} kWh。日期、144时段、阶段时点、跨日SOC、U/R非负互斥、累计承诺及紧急区间全部核对。',
      '',f'本次回归执行状态：{json.dumps(tests,ensure_ascii=False)}。not_packaged表示公开衍生包不含原始冻结校验器，未进行该项核验；空记录表示此次未运行回归。可用接手快照检查条目为{protected}，为0时不代表检查了私有原件。日志保存在Q3，不写入Q1/Q2。Excel逐值回读由Q3输出测试执行，其本次状态以上述记录为准。',
      '','## WARNING与结论边界','',
      '1. 主A不退费并另收调减违约费不是唯一官方解释；B与B_net是分开的替代实验。A无调减符合支配逻辑。',
      '2. 时间结束点、效率各90%、十分钟实时反馈均为继承假设；不能写成题目逐字规定。',
      '3. Q2和Q3正式期初SOC不同；现金和末态调整费用同时披露，后者不是账单收入。',
      '4. 8子集共享正式配置日历；独立仅0时历史选择作为另一个基准。逐阶段新预报屏蔽保持调整权限，用于区分新PV信息和SOC反馈。',
      '5. estimated VOI是事前模型估计，realized_value是余日冻结对照，多个余日重叠不可直接加总年度收益。',
      '6. 更少紧急电量不代表更少紧急时段；正式Q3存在更多小缺口时段。',
      '7. 审计代码与求解器在本次同一任务中编写，但验算路径独立，不是外部第三方认证；数学模型仍仅为有限策略类中的历史选择。',
      '','这是公开衍生包的数值验收，不是私有原始冻结件的完整性证明，也不包含人工视觉审查。未改动任何Q1/Q2既有日志或结果。']
    (OUT/'Q3最终验收审计.md').write_text('\n'.join(lines)+'\n');print('Independent audit PASS',float(total),flush=True)
    return result

if __name__=='__main__':audit()
