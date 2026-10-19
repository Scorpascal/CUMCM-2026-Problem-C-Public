"""All CSV, scientific plots, paper and official workbook payload generation."""
import os,shutil,subprocess
from dataclasses import replace
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from q4_utils import *
from q4_model import Config,CANDIDATES
from q4_features import Factory
from q4_risk import joint_paths
from microgrid_online import PERIODS,merge_emergency,terminal_value

WANTED=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in row)+'|' for row in rows])+'\n'
def fmt(x):return f'{float(x):,.4f}'

def exports(kind,dates,a,p,s):
    daily=[];dispatch=[];ledger=[];predrows=[];residualrows=[]
    for d in range(31,365):
        date=str(dates[d]);g0=s['versions'][d,0];g=s['G'][d]
        daily.append(dict(date=date,config=s['config'][d],initial_kwh=s['E'][d,0],terminal_kwh=s['E'][d,-1],
            **{key+'_yuan':s[key][d].sum() for key in ('plan_cost','up_cost','penalty','refund','emergency_cost')},
            total_cost_yuan=s['cash'][d].sum(),plan_kwh=g0.sum(),effective_kwh=g.sum(),emergency_kwh=s['H'][d].sum(),
            emergency_periods=int((s['H'][d]>1e-7).sum()),increase_kwh=s['U'][d].sum(),decrease_kwh=s['R'][d].sum(),
            price_mean=p[d].mean(),price_forecast_mae=np.abs(s['price_forecasts'][d,0,:144]-p[d]).mean()))
        for t in range(144):
            row=dict(date=date,index=t,period=PERIODS[t],load_kw=a[d,t,0]*6,pv_kw=a[d,t,1]*6,price_realized=p[d,t],
                price_forecast_0=s['price_forecasts'][d,0,t],G0=g0[t],G6=s['versions'][d,1,t],G12=s['versions'][d,2,t],G18=s['versions'][d,3,t],G_effective=g[t],
                C_real=s['C'][d,t],D_real=s['D'][d,t],H_real=s['H'][d,t],W_real=s['W'][d,t],E_start=s['E'][d,t],E_end=s['E'][d,t+1],
                **{key+'_yuan':s[key][d,t] for key in ('plan_cost','up_cost','penalty','refund','emergency_cost','cash')})
            dispatch.append(row)
        for k in range(4 if kind==43 else 1):
            n=144-k*36;now=d*144+k*36
            for j in range(n):
                t=k*36+j
                predrows.append(dict(kind=kind,issue_date=date,issue_hour=k*6,training_last_index=now-1,target_date=date,index=t,lead_minutes=(j+1)*10,
                    price_forecast=s['price_forecasts'][d,k,j],price_realized=p[d,t],load_forecast_kw=s['predictions'][d,k,j,0]*6,pv_forecast_kw=s['predictions'][d,k,j,1]*6))
                residualrows.append(dict(kind=kind,issue_index=now,target_end_index=d*144+t+1,lead=j+1,
                    load_error=a[d,t,0]-s['predictions'][d,k,j,0],pv_error=a[d,t,1]-s['predictions'][d,k,j,1],
                    price_error=p[d,t]-s['price_forecasts'][d,k,j],available_after=d*144+t+1))
                if k:
                    u=s['U'][d,k-1,t];r=s['R'][d,k-1,t]
                    ledger.append(dict(decision_date=date,stage=k,decision_hour=k*6,delivery_date=date,index=t,period=PERIODS[t],
                        previous_commitment=s['versions'][d,k-1,t],new_commitment=s['versions'][d,k,t],increase_U=u,decrease_R=r,
                        price_forecast=s['price_forecasts'][d,k,j],price_realized=p[d,t],up_cost=1.5*p[d,t]*u,penalty=.5*p[d,t]*r,refund=0,
                        adjustment_charge=p[d,t]*(1.5*u+.5*r)))
    write_csv(OUT/f'q{kind}_daily_summary.csv',daily);write_csv(OUT/f'q{kind}_dispatch_10min.csv',dispatch)
    write_csv(OUT/f'q{kind}_price_forecasts.csv',predrows);write_csv(OUT/f'q{kind}_price_residuals.csv',residualrows)
    if ledger:write_csv(OUT/f'q{kind}_adjustment_ledger.csv',ledger)
    merged=merge_emergency(dates[31:],s['H'][31:]);write_csv(OUT/f'q{kind}_emergency_purchase_merged.csv',merged)
    write_csv(OUT/f'q{kind}_warmup_january.csv',[dict(date=str(dates[d]),initial=s['E'][d,0],terminal=s['E'][d,-1],cash=s['cash'][d].sum()) for d in range(31)])
    selected=[r for r in daily if r['date'] in WANTED]
    write_csv(OUT/f'selected_days/q{kind}_summary.csv',selected)
    for date in WANTED:
        d=next(i for i,v in enumerate(dates) if str(v)==date)
        write_csv(OUT/f'selected_days/q{kind}_{date}_table1.csv',[dict(period=PERIODS[t],G0=s['versions'][d,0,t],effective=s['G'][d,t]) for t in (60,72,84,96,108,120)])
        write_csv(OUT/f'selected_days/q{kind}_{date}_table2.csv',[dict(period=f'{j*4}:00-{(j+1)*4}:00',charge=s['C'][d,j*24:(j+1)*24].sum(),discharge=s['D'][d,j*24:(j+1)*24].sum()) for j in range(6)])
        write_csv(OUT/f'selected_days/q{kind}_{date}_dispatch.csv',[r for r in dispatch if r['date']==date])
    payload={};head=['日期\\时间']+PERIODS+['全天购电量','全天购电费']
    payload['计划购电量']=[head]+[[str(dates[d])]+s['versions'][d,0].tolist()+[float(s['versions'][d,0].sum()),float(s['cash'][d].sum() if kind==42 else s['plan_cost'][d].sum())] for d in range(31,365)]
    if kind==43:payload['调整购电量']=[head]+[[str(dates[d])]+s['G'][d].tolist()+[float(s['G'][d].sum()),float(s['cash'][d].sum())] for d in range(31,365)]
    payload['充放电量']=[['日期','时间段','充电量','放电量','时刻','储电量']]
    for d in range(31,365):
        for j in range(6):payload['充放电量'].append([str(dates[d]) if j==0 else None,f'{j*4}:00-{(j+1)*4}:00',float(s['C'][d,j*24:(j+1)*24].sum()),float(s['D'][d,j*24:(j+1)*24].sum()),'0:00' if j==0 else '24:00' if j==1 else None,float(s['E'][d,0 if j==0 else -1]) if j<2 else None])
    payload['紧急购电量']=[['日期','购电时间段','购电量']]
    for date in dates[31:]:
        group=[r for r in merged if r['date']==str(date)]
        if group:
            for j,r in enumerate(group):payload['紧急购电量'].append([str(date) if j==0 else None,r['period'],r['kwh']])
        else:payload['紧急购电量'].append([str(date),'无',0.])
    write_json(OUT/f'q{kind}_workbook_payload.json',payload)
    return daily,merged

def figures(kind,dates,a,p,s,daily):
    out=OUT/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=145);fig.savefig(out/f'{name}.svg');plt.close(fig)
    extreme={int(p[31:].mean(1).argmax())+31,int(np.ptp(p[31:],axis=1).argmax())+31,int(np.argmax(s['cash'][31:].sum(1)))+31}
    extreme.add(int(np.unravel_index(p[31:].argmax(),p[31:].shape)[0])+31)
    extreme.add(next(i for i,d in enumerate(dates) if str(d)==max(daily,key=lambda r:r['price_forecast_mae'])['date']))
    chosen={i for i,d in enumerate(dates) if str(d) in WANTED}|extreme
    for d in sorted(chosen):
        x=np.arange(144)/6;fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
        axes[0].plot(x,p[d],color='black',label='Realized price');axes[0].plot(x,s['price_forecasts'][d,0,:144],label='00:00 forecast');axes[0].set_ylabel('CNY/kWh')
        if kind==43:
            for k in (1,2,3):axes[0].plot(x[k*36:],s['price_forecasts'][d,k,:144-k*36],label=f'{6*k:02}:00 forecast',lw=.7)
        axes[1].plot(x,(a[d,:,0]-a[d,:,1])*6,label='Actual net load',color='black',lw=1)
        axes[1].step(x,s['versions'][d,0]*6,label='Original G0',lw=.8);axes[1].step(x,s['G'][d]*6,label='Effective G',lw=.9)
        axes[1].fill_between(x,0,s['H'][d]*6,alpha=.4,label='Emergency');axes[1].set_ylabel('kW')
        axes[2].plot(np.arange(145)/6,s['E'][d],label='Stored energy',color='#176b63');axes[2].axhline(1200,color='gray',ls=':');axes[2].axhline(10800,color='gray',ls=':');axes[2].set_ylabel('kWh')
        axes[3].plot(x,np.cumsum(s['cash'][d]),label='Cumulative cash',color='#923b36');axes[3].set(ylabel='CNY',xlabel='Hour')
        for ax in axes:ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.15)
        fig.suptitle(f'Q4-{kind-40} / {dates[d]}');save(fig,f'q{kind}_{dates[d]}')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for key,label in [('plan_cost','Plan'),('up_cost','Increase'),('emergency_cost','Emergency')]:
        vals=[sum(s[key][d].sum() for d in range(31,365) if dates[d].month==m) for m in range(2,13)]
        ax.plot(range(2,13),np.array(vals)/1e6,'o-',label=label)
    ax.set(xlabel='Month',ylabel='Million CNY',title=f'Q4-{kind-40} monthly cash');ax.legend();save(fig,f'q{kind}_monthly_costs')
    candidates=read_csv(OUT/f'experiments/q{kind}_candidate_summary.csv')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for r in candidates:
        joint='_joint_' in r['config'];ax.scatter(float(r['daily_mean_yuan']),float(r['worst5_mean_yuan']),c='#ae4741' if joint else '#27756a',s=35)
    ax.set(xlabel='Mean daily cash (CNY)',ylabel='Worst 5% mean daily cash (CNY)',title=f'Q4-{kind-40} fixed candidates (post-hoc diagnostics)');save(fig,f'q{kind}_risk_pareto')
    write_json(OUT/f'experiments/q{kind}_extreme_days.json',dict(max_mean_price=str(dates[np.argmax(p[31:].mean(1))+31]),max_range_price=str(dates[np.argmax(np.ptp(p[31:],axis=1))+31]),max_single_price=str(dates[np.unravel_index(p[31:].argmax(),p[31:].shape)[0]+31]),max_forecast_error=max(daily,key=lambda r:r['price_forecast_mae'])['date'],max_cost=max(daily,key=lambda r:r['total_cost_yuan'])['date']))

def documents(all_results,all_daily,all_merged):
    summaries={k:json.loads((OUT/f'q{k}_summary.json').read_text()) for k in (42,43)}
    comparisons={k:json.loads((OUT/f'experiments/q{k}_comparisons.json').read_text()) for k in (42,43)}
    metrics=read_csv(OUT/'price_metrics.csv');stats=json.loads((OUT/'price_statistics.json').read_text())
    text='# 问题4：波动电价下的因果购电与滚动调整\n\n'
    text+='## 1 递进与结论范围\n\n原题要求分别重算问题2、3；本文继承已冻结的物理与权限，增加因果价格预测。题面没有明确全天价预先公布，因此未来未知是本文新增解释，Oracle-P为替代信息条件。所有正式现金按附件4交付十分钟实际价结算。模型是预注册有限策略类中的历史经济选择，不宣称全局随机最优。\n\n'
    delta42=summaries[42]['total_cost_yuan']-comparisons[42]['frozen_repriced']['total_cost_yuan']
    delta43=summaries[43]['total_cost_yuan']-comparisons[43]['frozen_repriced']['total_cost_yuan']
    text+=f'关键限制：正式Q4-2比冻结Q2动态重计价多支出{delta42:,.4f}元；正式Q4-3比冻结Q3动态重计价费用变化为{delta43:,.4f}元（负值为节省）。因此不能宣称新增价格预测在两问都创造年度收益，更不能宣称复杂方法普遍优越。正式采用的是事前锁定候选、月首历史经济选择规则，未依据全年已知结果回填最优模型。\n\n'
    text+='## 2 价格数据与样本外预测\n\n'+(OUT/'附件4结构审计.md').read_text().split('\n',1)[1]+'\n'
    text+=(OUT/'候选价格预测模型.md').read_text().split('\n',1)[1]+'\n'
    text+=table(['价格模型','0时MAE','RMSE','rMAE（昨日）','bias'],[[r['model'],fmt(r['mae']),fmt(r['rmse']),fmt(r['rmae']),fmt(r['bias'])] for r in metrics if r['stage']=='0' and r['group']=='all'])
    text+='rMAE分母是同一测试点上的昨日同刻基线MAE，不是Lago文中优先推荐的7日前基线。高价分组阈值用于事后诊断，不进入预测。概率区间基于已完成历史误差，不采用Conformal，不保证覆盖率。详细月份、时刻、时距和高低价分组见price_metrics.csv。\n\n'
    text+=table(['预测器','名义覆盖','实际覆盖','平均宽度','pinball'],[[r['model'],r['nominal'],fmt(r['coverage']),fmt(r['width']),fmt(r['pinball'])] for r in read_csv(OUT/'price_interval_metrics.csv')])
    text+='![四季价格预测](figures/price_forecast_precheck.png)\n\n## 3 数学模型、价格加权风险、场景与执行\n\n'
    text+=(OUT/'候选数学模型.md').read_text().split('\n',1)[1]
    text+='\n## 4 正式历史选择与全年费用\n\n'
    text+=table(['指标','Q4-2','Q4-3'],[[key,fmt(summaries[42][key]),fmt(summaries[43][key])] for key in ('plan_cost_yuan','up_cost_yuan','penalty_yuan','emergency_cost_yuan','total_cost_yuan','terminal_adjusted_cost_yuan','emergency_kwh','emergency_periods','increase_kwh','decrease_kwh','initial_kwh','terminal_kwh','worst5_mean_yuan','high_price_emergency_kwh')])
    saving=summaries[42]['total_cost_yuan']-summaries[43]['total_cost_yuan']
    text+=f'Q4-3相对Q4-2现金节省{saving:,.4f}元，比例{saving/summaries[42]["total_cost_yuan"]:.4%}。二者信息与计划调整权限不同，本次正式期初库存分别为{summaries[42]["initial_kwh"]:.4f}和{summaries[43]["initial_kwh"]:.4f}kWh，期末库存见表，不能全部解释为单一预报价值。终端调整使用1月已知价格平均日型最低24点均值/.9，属于辅助比较，不是实际收入。\n\n'
    for kind in (42,43):
        selected=[r for r in read_csv(OUT/f'experiments/q{kind}_monthly_selection.csv') if r['selected']=='True']
        text+=f'### Q4-{kind-40}月度选择\n\n'+table(['月首日索引','配置','历史起止','验证评分'],[[r['selection_day'],r['config'],r['begin']+'–'+r['end'],fmt(r['score'])] for r in selected])
        text+='配置名E0为冻结反馈，E12/E36为2/6小时储能MPC；q=0是无风险裕度。阶段提案和VOI使用E0代理，E1是否采用由完整真实执行历史费用检验。\n\n'
        comp=comparisons[kind];formal=summaries[kind]['total_cost_yuan']
        text+=f'### Q4-{kind-40}对照与消融\n\n'+table(['对照','真实总费用','相对正式多支出','期初E','末E'],[[name,fmt(r['total_cost_yuan']),fmt(r['total_cost_yuan']-formal),fmt(r.get('initial_kwh',0)),fmt(r.get('terminal_kwh',0))] for name,r in comp.items()])
        text+=f'冻结Q{kind-40}按附件4重计价与正式差额为{comp["frozen_repriced"]["total_cost_yuan"]-formal:,.4f}元。该基准保持原G/U/R/C/D/H轨迹，绝不反向改写Q2/Q3。Oracle-P保持负载/PV预测、只揭示当天完整价格，差额不保证非负，不能称为严格下界。Full Oracle放松信息且以正式期首末SOC求全48,096点LP，才是相同边界的费用下界。\n\n'
        text+='Oracle-NP仅揭示负载与PV（最长48小时），价格仍因果预测；两个单类Oracle均将已揭示变量的场景残差置零，不再给确定变量添加人工不确定性。它们沿正式选定配置日历重放，而非重新优化全部算法。决策锁价把当时最近已完成价格用于未来普通交易，改变了账单定义，其低费用不代表同一市场的算法优势。\n\n'
        text+=f'配对E1相对E0现金变化（正为节省）：{comp["paired_E0"]["total_cost_yuan"]-comp["paired_E1"]["total_cost_yuan"]:,.4f}元；联合场景相对同预测E0：{comp["paired_E0"]["total_cost_yuan"]-comp["paired_joint"]["total_cost_yuan"]:,.4f}元；CVaR相对风险中性：{comp["paired_joint"]["total_cost_yuan"]-comp["paired_CVaR"]["total_cost_yuan"]:,.4f}元。分别是闭环总效应，不能全部归因于当前第一步。\n\n'
        text+=table(['配对方案','均日费','最差5%日均费','高价紧急kWh'],[[name,fmt(comp[name]['daily_mean_yuan']),fmt(comp[name]['worst5_mean_yuan']),fmt(comp[name]['high_price_emergency_kwh'])] for name in ('paired_E0','paired_E1','paired_joint','paired_CVaR','independent_price')])
        text+='联合场景只在有限q提案中优化。价格残差跨场景循环重排是独立近似压力对照，不是证明真实变量独立的统计检验。CVaR和场景规模结果只适用于本策略类；DRO与GBDT没有实施，不能声称它们一定无效。\n\n'
        text+=f'固定正式承诺序列的纯执行诊断：E36相对E0节省{comp["fixed_commitment_E0"]["total_cost_yuan"]-comp["fixed_commitment_E36"]["total_cost_yuan"]:,.4f}元。两者只使用当时已发行版本的未来G，不能看到后续调整；该诊断冻结了外生承诺，因此也不能替代完整闭环选参。\n\n'
        candidates=read_csv(OUT/f'experiments/q{kind}_candidate_summary.csv')
        text+=table(['场景/风险/时域固定候选','总现金','最差5%日均费'],[[r['config'],fmt(r['total_cost_yuan']),fmt(r['worst5_mean_yuan'])] for r in candidates if '_joint_' in r['config'] or '_R48_' in r['config']])
        text+=table(['同q=.65/E0价格模型','事后费用','MAE'],[[r['config'].split('_q')[0],fmt(r['total_cost_yuan']),fmt(next(z['mae'] for z in metrics if z['stage']=='0' and z['group']=='all' and z['model']==r['config'].split('_q')[0]))] for r in candidates if '_q0.65_ordinary_' in r['config'] and r['config'].endswith('E0')])
        text+='费用不仅取决于MAE，还取决于错误出现时的购电量、SOC和紧急倍价。以上全年固定候选排名是正式选择完成后的诊断，不能用其最低项替换月度历史日历。\n\n'
        text+=f'![月度费用](figures/q{kind}_monthly_costs.png)\n\n![尾部比较](figures/q{kind}_risk_pareto.png)\n\n'
        text+='尾部散点中红色为联合场景候选（含CVaR），绿色为其他候选。横纵轴分别为平均日现金与最差5%日平均现金，均越低越好；图是全部固定候选的事后诊断。\n\n'
    events=[r for r in read_csv(OUT/'q43_voi_events.csv') if int(r['day'])>=31]
    local=read_csv(OUT/'q43_voi_realized_local.csv')
    text+='### 阶段调整价值\n\n'+table(['阶段','UPDATE事件数','平均事前代理VOI','平均局部真实现金价值'],[[str(k*6)+':00',sum(r['accepted']=='True' for r in events if int(r['stage'])==k),fmt(np.mean([float(r['voi']) for r in events if int(r['stage'])==k])),fmt(np.mean([float(r['realized_E0_proxy_cash_value']) for r in local if int(r['stage'])==k]))] for k in (1,2,3)])
    text+='局部真实值已扣本次调整费，但保持承诺至日末、使用E0反馈，阶段余日区间重叠，不能相加解释全年收益。新价格屏蔽与新PV屏蔽的全年对照见上表。\n\n'
    text+='## 5 四个指定日期与价格极端日\n\n'
    for kind in (42,43):
        selected=[r for r in all_daily[kind] if r['date'] in WANTED]
        text+=f'### Q4-{kind-40}指定日\n\n'+table(['日期','原计划费','调增费','紧急费','总费','紧急kWh','日初/末E'],[[r['date'],fmt(r['plan_cost_yuan']),fmt(r['up_cost_yuan']),fmt(r['emergency_cost_yuan']),fmt(r['total_cost_yuan']),fmt(r['emergency_kwh']),fmt(r['initial_kwh'])+'/'+fmt(r['terminal_kwh'])] for r in selected])
        for date in WANTED:
            text+=f'#### {date}\n\n'
            text+='表1（普通购电，紧急另列）\n\n'+table(['时段','0时原计划kWh','有效普通kWh'],[[r['period'],fmt(r['G0']),fmt(r['effective'])] for r in read_csv(OUT/f'selected_days/q{kind}_{date}_table1.csv')])
            text+='表2（真实充放电）\n\n'+table(['时段','充电kWh','放电kWh'],[[r['period'],fmt(r['charge']),fmt(r['discharge'])] for r in read_csv(OUT/f'selected_days/q{kind}_{date}_table2.csv')])
            h=[r for r in all_merged[kind] if r['date']==date]
            text+='表3（连续紧急购电）\n\n'+table(['时间段','购电kWh'],[[r['period'],fmt(r['kwh'])] for r in h] or [['无','0']])
            text+=f'![调度](figures/q{kind}_{date}.png)\n\n'
        extremes=json.loads((OUT/f'experiments/q{kind}_extreme_days.json').read_text())
        text+='极端日期自动选择：'+str(extremes)+'。其中价格最高不必对应费用最高，需同时观察负载、PV、承诺和SOC。\n\n'
        for key in ('max_mean_price','max_range_price','max_single_price','max_forecast_error','max_cost'):
            date=extremes[key];r=next(r for r in all_daily[kind] if r['date']==date)
            text+=f'{key}={date}，日均价{r["price_mean"]:.4f}，总费{r["total_cost_yuan"]:,.4f}元，紧急{r["emergency_kwh"]:,.4f}kWh，首末SOC{r["initial_kwh"]:.2f}/{r["terminal_kwh"]:.2f}。\n\n'
            text+=f'![极端日调度](figures/q{kind}_{date}.png)\n\n'
    text+='## 6 验证、局限与文件口径\n\n'+(OUT/'输出与费用口径.md').read_text().split('\n',1)[1]+'\n'
    text+='本模型用一年数据，月首21日评分可能受短窗噪声影响；场景少时CVaR近似最大样本；风险裕度、终端代理与执行反馈均是模型设计。当前时段供需均值是聚合反馈假设。更新门控只保证当时代理评价不劣，不能保证真实费用单调下降。部分替代沿正式配置日历重放，没有各自重新调参。价格误差区间及日费用尾部是经验描述，不是概率保证。最终验证结果见Q4最终验收审计.md；哈希完整性不替代数学模型正确性。\n\n'
    text+='q43_voi_events.csv保存发行时KEEP/UPDATE代理评分；q43_voi_realized_local.csv事后以相同当时SOC，将旧/新承诺分别保持至日末，用真实价格和E0反馈评价局部现金差，已经扣除本次调整费。该反事实不采用后续再次调整，且多个阶段余日区间重叠，不能加总成年收益，也不是E1精确VOI。流程记录：正式滚动前完成价格预检与27项核心测试，Excel全量验算在正式数组生成后完成；不声称Excel验算先于全年求解。\n\n'
    text+='## 7 文献依据\n\n'+(OUT/'文献方法映射.md').read_text().split('\n',1)[1]
    (OUT/'Q4_final.md').write_text(text)
    # Decision report retains the core interpretation, numeric choices and all comparison answers.
    start=text.index('## 4 正式历史选择');end=text.index('## 5 四个指定')
    decision='# Q4最终决策报告\n\n'+text[text.index('## 1 '):text.index('## 3 ')]+text[start:end]
    decision+='\n四指定日完整表1/2/3、极端日及模型公式见Q4_final.md。无DRO/GBDT；概率区间是经验诊断，非Conformal。价格加权、联合场景和CVaR是否主用由逐月配置逐项说明。未来价格未知、当前报价可用于E1及交付价结算是新增/继承解释，不是原题唯一官方规定。\n\n核心测试和全量回读、旧问回归、冻结保护最终状态见Q4最终验收审计.md。\n'
    (OUT/'Q4最终决策报告.md').write_text(decision)

def deliver():
    from literature_q4 import generate
    from supplement_q4 import supplement
    generate();supplement()
    writable();dates,a,fixed,p,records=read_inputs();f=Factory(a,p,records,fixed)
    results={};daily={};merged={};scene_rows=[]
    for kind in (42,43):
        s=dict(np.load(OUT/f'experiments/q{kind}_formal.npz'));results[kind]=s
        daily[kind],merged[kind]=exports(kind,dates,a,p,s);figures(kind,dates,a,p,s,daily[kind])
        cal=json.loads((OUT/f'experiments/q{kind}_calendar.json').read_text())
        for date in WANTED:
            d=next(i for i,v in enumerate(dates) if str(v)==date);conf=Config(**cal[d])
            for k in range(4 if kind==43 else 1):
                ff=f.feature(kind,d,k,conf.price_model);n=144-k*36;paths,ps,ix=joint_paths(ff.prediction,ff.price_forecast[:n],ff.residual,conf.scenarios)
                for j,origin in enumerate(ix):
                    for t in range(n):scene_rows.append(dict(kind=kind,issue=date,stage=k,scenario=j,historical_origin=ff.origins[origin] if origin>=0 else -1,target_offset=t,
                        load_kwh=paths[j,t,0],pv_kwh=paths[j,t,1],price_scenario=ps[j,t],probability=1/len(ix),historical_target_end=ff.origins[origin]*144+k*36+288 if origin>=0 else -1))
    write_csv(OUT/'joint_scenarios.csv',scene_rows)
    documents(results,daily,merged)
    (OUT/'README.md').write_text('''# 问题4完整交付

正式文件result4-2.xlsx与result4-3.xlsx；论文Q4_final.md；模型选择Q4最终决策报告.md；验收Q4最终验收审计.md。

本目录是公开衍生副本。先按数据说明自行准备附件1–4及附件5的五份结果模板；在code目录执行 `python reproduce.py --through Q4 --check-inputs` 检查文件存在，再执行 `python reproduce.py --through Q4`。入口顺序生成Q1/Q2/Q3依赖，再执行Q4准备、预测预检、核心测试、两类全年历史选择、固定候选诊断、敏感性、CSV/图/论文与工作簿、数值审计。Q4可分步使用--prepare、--precheck、--compute-kind 42/43、--diagnostics-kind 42/43、--deliver。全部模型计算与Excel填写使用公开Python依赖，export_workbooks_public.py为默认导出器，不需要Node。

使用numpy/scipy/openpyxl/matplotlib求解和回读；依赖范围见公开仓库config/requirements.txt。--diagnostics-kind还需要Q2和Q3各自的experiments/formal_arrays.npz，并可复用当前副本已完成的Q4候选NPZ。若修改算法，必须在新独立副本重新计算其数值结果，不能把旧数组作为本次重算证据。

正式价格未知、当前执行报价可观察是假设；主A不退款、逐事件上一承诺、按交付实际价结算。Q42仅0时计划；Q43只0/6/12/18调整。1月预运行，2–12月334日正式。各策略SOC独立连续。相同主日历的敏感性不等于各自重新调优；场景属于有限策略SAA，不声称全局随机最优。

audit_q4.py从底层CSV和原始输入复算，再全量回读Excel并跑Q1/Q2/Q3/Q4测试；Q3交易审计在临时独立副本运行，日志仅保存在Q4。原始冻结核验器未随公开包分发，相关检查记录为not_packaged。数值通过不等于人工视觉审查或原始私有档案的完整性证明。

请将公开Release包解压到独立目录后核验其公开清单；不要用原始冻结清单检查此衍生包。完整重算会写入该副本的Q1/Q2/Q3/Q4目录。公开包不包含原始ZIP、冻结清单、接手快照或精确私有环境记录；原始档案继续单独保留。

最终模型详见q4_model.py，价格预测见q4_price_forecast.py，信息集及历史场景见q4_features.py/q4_risk.py，执行见q4_execution_mpc.py，CSV/图/论文见report_q4.py，默认工作簿填写代码见export_workbooks_public.py。literature_q4.py保留文献映射生成代码；supplement_q4.py生成固定承诺执行和局部VOI诊断。Oracle已揭示变量的残差必须为零，相关对照由solve_q4.py的诊断流程生成。
''')
    from export_workbooks_public import export
    export()

if __name__=='__main__':deliver()
