"""Generate Gate A/B evidence and locked protocol before optimization."""
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import numpy as np
from q4_utils import *
from q4_model import CANDIDATES

def prepare():
    writable();dates,a,fixed,p,records=read_inputs()
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidates=[asdict(c) for c in CANDIDATES],
        selection='monthly previous 21 completed days; common validation initial SOC; cash minus historical terminal inventory proxy',
        official_period='2025-02-01/2025-12-31',startup='Jan 1 G=0 battery idle; Jan 2-31 mean7 q=.8, independent Q42/Q43 SOC',
        price_information='strictly before stage; current price only for E1 current aggregate action',
        risk='finite shared-G candidates, causal frozen feedback scenario evaluation; no free scenario-specific future optimization',
        execution='E1 is independently tested execution challenger; stage proposal/VOI uses E0 proxy and historical selection evaluates actual E1',
        scenarios='synchronized complete 48h residuals, <=30 origins, deterministic evenly spaced subsample',
        price_floor=.001,ridge_penalty=10.,price_stage_decay_periods=12,
        sensitivity_predeclared=['Oracle-P today only','Oracle-NP','full same-boundary LP','frozen Q2/Q3 repricing','E0/E1 paired',
          'joint price residual permutation','B/B_net','decision-time lock uses last completed price','only0','stale_price','stale_pv','current-price lag','Q42 official PV0'],
        optional_declined=['DRO','GBDT','conformal formal predictor'],no_formal_results_seen=True)
    target=OUT/'experiments/protocol.json'
    if target.exists():
        old=json.loads(target.read_text());assert old['candidates']==protocol['candidates']
    else:write_json(target,protocol)
    stats=dict(shape=list(p.shape),count=p.size,min=float(p.min()),max=float(p.max()),mean=float(p.mean()),std=float(p.std()),
      quantiles={str(q):float(np.quantile(p,q)) for q in (0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1)},
      negative=int((p<0).sum()),zero=int((p==0).sum()),missing=int((~np.isfinite(p)).sum()),
      acf={str(lag):float(np.corrcoef(p.ravel()[:-lag],p.ravel()[lag:])[0,1]) for lag in (1,6,36,144,1008)},
      mean_profile_fixed_mae=float(np.abs(p.mean(0)-fixed).mean()),mean_profile_fixed_corr=float(np.corrcoef(p.mean(0),fixed)[0,1]),
      adjacent_abs_jump_max=float(np.abs(np.diff(p.ravel())).max()),interday_same_slot_mae=float(np.abs(np.diff(p,axis=0)).mean()),
      max_mean_day=str(dates[int(p.mean(1).argmax())]),max_range_day=str(dates[int(np.ptp(p,axis=1).argmax())]),
      max_single_day=str(dates[int(np.unravel_index(p.argmax(),p.shape)[0])]))
    write_json(OUT/'price_statistics.json',stats)
    write_csv(OUT/'price_monthly.csv',[dict(month=m,mean=float(p[[d.month==m for d in dates]].mean()),std=float(p[[d.month==m for d in dates]].std())) for m in range(1,13)])
    write_csv(OUT/'price_weekday.csv',[dict(weekday=k,mean=float(p[[d.weekday()==k for d in dates]].mean())) for k in range(7)])
    write_csv(OUT/'price_profile.csv',[dict(index=t,fixed=fixed[t],mean=p[:,t].mean(),median=np.median(p[:,t]),q10=np.quantile(p[:,t],.1),q90=np.quantile(p[:,t],.9),std=p[:,t].std()) for t in range(144)])
    write_csv(OUT/'price_daily.csv',[dict(date=str(d),mean=p[j].mean(),min=p[j].min(),max=p[j].max(),range=np.ptp(p[j])) for j,d in enumerate(dates)])
    write_csv(OUT/'time_mapping.csv',[dict(date=str(d),index=t,interval_start=f'{t//6}:{t%6*10:02}',interval_end=f'{(t+1)//6}:{(t+1)%6*10:02}',price=p[j,t]) for j,d in enumerate(dates) for t in range(144)])
    docs={
    '题意与价格信息集审计.md':'''# 题意与价格信息集审计

## 原题明文
问题4在波动外网电价下重新计算问题2、3，附件4给出全年电价。紧急购电是交易时刻电价5倍；Q3调增1.5倍、调减50%违约。没有明文规定0时公布全天价格，没有定义价格预报发布机制，也未明示调整下单时与交付时哪个锁价。

## 附件能够确认
附件4有365日×144点实现价格及时间标签，不带发行时间或预测版本。附件5要求两套334日结果。附件3是光伏预报，不能作为电价预报。

## 继承Q2/Q3冻结口径
Q42无官方PV预报、0时普通计划当天固定；Q43有0/6/12/18官方PV，只有这些节点可修改未执行承诺。主A不退款，针对上一有效承诺逐次结算，按交付十分钟价格计费。G/H/C/D/W母线侧kWh，效率各.9，SOC1200–10800、功率5000 kW，跨日连续。实时供需均值是既有反馈聚合假设。

## 本文新增解释
主方案按实时揭示解释：阶段前只知已完成价格，预测未来。计划和调整只用预测/场景价，账单统一用附件4实际交付价。6时调整18时交付用18时预测计算VOI，之后用18时实际价结算。E0不读价格；E1假设当前时段报价可用于当前反馈动作，但后续报价不可见；另检验仅上一完成区间价格可用。E1只改变储能与缺口H，不改变G，不用紧急电充电。

## 存在歧义的替代解释
Oracle-P当日完整价已知、负载/PV仍预测，属于信息敏感性，不是唯一官方解释或严格全局下界。决策锁价敏感性使用决策前最后完成价格作为可观察代理；它不是题面给出的真实订单价格。Q42+0时官方PV仅信息增强敏感性。B/B_net继承Q3的独立替代结算，绝不混入正式A。
''',
    '附件4结构审计.md':f'''# 附件4结构审计

实际读取并断言全年连续日期与附件2完全相等，144个结束标签为0:10至0:00+1，无重复、缺日、缺值或非有限值；全部为正。精确统计见price_statistics.json。

价格最小{p.min():.4f}、最大{p.max():.4f}、均值{p.mean():.8f}、标准差{p.std():.8f}元/kWh。价格不乘1/6。数据结束点解释继承既有假设，输出time_mapping.csv保存52,560条显式映射。

附件1与附件4平均日型MAE={stats['mean_profile_fixed_mae']:.8f}，相关系数={stats['mean_profile_fixed_corr']:.8f}。相邻十分钟最大绝对跳变={stats['adjacent_abs_jump_max']:.4f}；相邻日同刻平均绝对变化={stats['interday_same_slot_mae']:.8f}。同刻均值接近并不意味着每天峰谷完全相同。

price_profile.csv记录均值、中位数、10/90分位、波动率与附件1；price_monthly/weekday/daily分别保留月份、星期、每日均值峰谷差；ACF记录1/6/36/144/1008步。描述统计使用全年仅作事后说明，不能用于正式预测参数和历史选择。
''',
    '候选价格预测模型.md':'''# 候选价格预测模型

P0昨日同刻；P1近7日同刻均值；P2近期与最多4个同星期日各半；P3过去90日工作日类型同刻中位数；P4滚动30日Ridge；P4E扩展窗口Ridge。全部在每个起点重新校准，训练目标严格早于起点。Ridge使用上一日同刻、7日前同刻、近7日同刻、日内一二谐波和周末标记，惩罚10，不使用外部数据或未来真实负载/PV。少于9日退回组合预测。

阶段新增已完成最近一小时价格偏差，用两小时指数衰减更新后续曲线。预测下界.001只为数值正性，不能由全年最小值倒推。无历史日公开回退.5且零G静置，不影响正式期直接预测。经验残差50/80/90%区间作为诊断，不声称分布无关覆盖保证。不引入GBDT、LSTM或Conformal复杂训练，无新增机器学习依赖。
''',
    '候选数学模型.md':r'''# 候选数学模型

## 共同物理和信息约束
原题流量G/H/C/D、状态E，继承剩余能量W；Q4新增真实价p、发行预测p_hat、场景p_s。G+PV+D+H=L+C+W，E'=E+.9C-D/.9，C,D在[0,5000/6]，E在[1200,10800]，G/H/W非负，C与D互斥。初值6000，跨日相接。Q42的G0仅可测于0时信息集；Q43的Gk只可测于阶段k信息，过去t<36k不变，Gnew=Gold+U-R。

## 价格加权风险推导
单时段J(G)=E[pG+5p(N-G)+]。连续内点J'=E[p]-5E[p 1(N>G)]。最优价格加权尾部占总价格质量的1/5；只有独立p、N时退化普通0.8分位。离散经验分布用按价格加权的逆CDF；储能耦合使其成为解释性近似。候选q=.65/.8及基线.7、无裕度0；margin=max(0,weighted_quantile(eL-ePV, max(.001,p_hat+ep),q))。

## Q42普通计划与Q43调整提案
0时LP：min sum(p_hat G)-vE_T。调整LP：min sum(1.5 p_hat U+.5 p_hat R)+sum(前瞻正常购电)-vE_T；原计划沉没不重复收费。净负荷为预测加风险裕度。T24到午夜；R48到发行后48h，仅当天成为承诺。v=当前发行未来144点最低24价均值/.9，为代理价值非收入。LP提案沿用已有消循环/互斥理论，实际执行单向。

## 联合经验场景与CVaR
不同Q42/Q43分别记录已发行的48h负载/PV/价格预测误差，完整实现后加入过去30条库。同步整条三变量残差一起抽取；4/8/12/20条等时间距代表路径，不用聚类。以q={0,.65,.8,.9}产生共享G提案，同一逐点反馈器揭示场景当前供需并算真实同型H/E。场景号不能进入动作，不采用未来自由储能全知补救。

J=mean(未来可影响现金-v(E_T-E_0))+beta CVaR_alpha(未来可影响现金)。CVaR=min_z[z+sum(pi*max(C_s-z,0))/(1-alpha)]，beta=0/.1/.2，alpha=.9/.95；离散小场景尾部可退化为最大场景，须披露。0时现金含普通与紧急，阶段含增量调整、前瞻正常与紧急，不含沉没原计划。KEEP和UPDATE用同一场景，VOI=Jkeep-Jupdate，含调整费不重复扣。有限提案策略类，不是原随机控制问题全局最优。

## 价格感知执行候选
固定G，当前观测净负荷覆盖首点，剩余仅合法预测。每十分钟最小化sum(5 p_hat H)-vE_T，约束充电不超过max(G-N,0)，放电不超过max(N-G,0)，H=N-G-D在缺口段、盈余段H=0，故不从紧急渠道充电。时域12/36点，只执行第一步，余量免费吸收不劣时直接返回等价反馈。阶段提案/VOI保留E0评价代理，历史候选回放实际执行E1，不能把代理VOI冒充E1精确价值。此明确近似降低嵌套计算。

## 选择与正式账本
26个配置预注册于protocol.json；每月首日仅前21个完成日、验证段共同已知起始SOC、候选独立连续回放。评分现金-v_hist(末初E差)，v_hist由选择前价格历史平均日型计算。月内配置固定、每日/阶段预测重校准。正式选择先于全年固定候选诊断。所有正式费用按交付实际p：Q42=sum(pG0+5pH)；Q43再加逐事件sum(1.5pU+.5pR)，零退款。
''',
    '输出与费用口径.md':'''# 输出与费用口径

两文件保留附件5官方表名、非时间表头和示意布局。原模板144个区间从0:10–0:20至0:00–0:10+1，存在整体偏移；继承Q2/Q3已冻结解释，修正标签为自然日0:00–24:00，不循环平移数值。Q42计划表=G0，全天量=sum G0，费用=普通+紧急。Q43计划表=G0及原始计划账单；调整表=最终有效G及完整实际现金费用，两费用不能相加。充放电/SOC为实际执行；紧急按H>1e-7日内连续合并，无事件日写无/0。显示四位，内部双精度。额外预测、场景和账本另存CSV，不添加官方sheet。

总W含未用计划电，PV弃用上下界max(W-G,0)、min(W,PV)。终端价值只放算法对照，绝不写入官方现金。全年从1月1日6000独立推进，2月起334日正式输出；日末不重置。
''',
    '方案审计.md':'''# 方案审计

继承物理、Q42/Q43权限与Q3主A。修正旧综合蓝图把48h/MPC/CVaR预设主用的表述。候选因果价格预测、加权风险、同步场景、CVaR和E1以历史经济选择检验。DRO/GBDT为可选，本次不引入，不能据此声称无效。

E1收益采用同一计划算法E0/E1完整回放比较，状态路径可反馈影响翌日计划，因此是执行政策改变的总闭环效应。另固定已生成承诺的执行对照用于纯执行诊断。新价格/PV屏蔽保留调整权限与SOC更新。Oracle-P不是严格下界；Full Oracle以各正式期首末SOC求48,096点确定性LP作为同边界下界。结算歧义单列敏感性，未来价格绝不随意混入正式信息。
''',
    '口径核验.md':'''# 口径核验

题面未唯一规定未来价格可知性、当前报价公布时点及调减退款。本次新增严格因果价格、当前E1可见报价假设；继承交付时段结算、不退款、逐次上一承诺，替代实验分开。价格乘电量不乘时间，功率先除6。Q42无附件3；Q43最新已发布PV。初始6000，1月预运行，2月SOC各自继承。预报误差库目标结束<=发行时刻，当前报价与当前聚合供需只影响当前执行，不回写同刻之前形成的普通计划。
'''}
    for name,body in docs.items():(OUT/name).write_text(body)
    print('Gate A/B ready:',len(CANDIDATES),'predeclared candidates',flush=True)

if __name__=='__main__':prepare()
