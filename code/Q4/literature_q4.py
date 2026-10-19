"""Reproducible source-to-method notes, based on the four local full texts."""
from q4_utils import *
def generate():
    source=ROOT/'参考资料/问题4相关论文/PDF原文'
    write_json(OUT/'literature_sources.json',{p.name:dict(sha256=digest(p),source=str(p.relative_to(ROOT))) for p in sorted(source.glob('*.pdf'))})
    (OUT/'文献方法映射.md').write_text('''# 文献方法映射

以下依据本地四篇PDF正文的模型、实验和局限整理；PDF页码包含封面，不把原文结果当成本题收益。原始PDF留在参考资料目录，literature_sources.json记录输入哈希。

## Zhang等，2018，A Stochastic MPC based Approach to Integrated Energy Management in Microgrids

Sustainable Cities and Society 41, 349–362；DOI: https://doi.org/10.1016/j.scs.2018.05.044 。本地Stochastic_MPC_Microgrids_2018.pdf，34页作者稿。第9页及场景部分将净负载和价格误差分别建模，独立生成并组合概率；场景削减的两个步骤不等于两阶段非前视决策树。模型包含柴油机、燃料电池、需求侧和买卖电，采用随机滚动预调度与实际信息到达后的修正。

第18–20页的滚动控制和当步实际修正启发Q4的信息时序；本题不引入发电机、售电或需求响应。本文同步抽取负载/PV/价格整条残差，以保留它们的历史相关性，与原文独立组合有明确差别。第29–30页的比较来自约4日实验，不能迁移为本题节省百分比。原文风险中性期望费用不是CVaR模型；其日末库存设定也不覆盖本题连续跨日要求。本文只在有限共享G提案类内进行场景评价，不能借文献声称解得完整随机最优。

## Hossain等，2021，Energy management of community energy storage in grid-connected microgrid under uncertain real-time prices

Sustainable Cities and Society 66, 102658；DOI: https://doi.org/10.1016/j.scs.2020.102658 。本地Community_Storage_Uncertain_Real_Time_Prices_2021_AAM.pdf共26页。必须保留版本警告：机构封面标记接受稿，但内页题名为Energy Management of Community Microgrids Considering Uncertainty using Particle Swarm Optimisation，并带2020年6月预印本及NOT PEER-REVIEWED水印，另列预印本DOI 10.20944/preprints202006.0205.v1。因此本次依据的是所存正文，不能断言与期刊最终版本逐字一致。

正文用PSO预调度和实时修正、正态误差Monte Carlo及少量代表场景，示例约48小时。其小时提前价格通知、无功率限制的外网交换、相同购售价、40kWh电池，均不同于本题。可借鉴“区分预测计划与实际执行”，不能照搬售电利润，也不能由其特定条件下负载/PV不确定性简化推出本题无需联合风险。本题5倍紧急电价和调整费用直接改变这种结论。该文不提供采用CVaR或DRO的直接依据。

## Alves等，2023，Battery Storage Systems Control Strategies with Intelligent Algorithms in Microgrids with Dynamic Pricing

Energies 16(14), 5262；DOI: https://doi.org/10.3390/en16145262 。本地Battery_Control_Dynamic_Pricing_2023.pdf共30页，期刊排版正文。非线性自回归ANN使用3个滞后、15个隐层节点，结合以SOC、PV、负载和价格为输入的Mamdani模糊控制。实验采用法国Nord Pool小时价格和晴/阴日24小时示例。

可以借鉴“电池是否放电需考虑价格与库存”的执行层思想，因此设置固定G的2/6小时MPC候选；本文并未复现其ANN/模糊系统。原文购售电和无本题调整费的设定不适用。第22页比较的期末库存存在显著差别，不能只比较现金就泛化方法优势；本文同时披露正式期首末SOC和库存价值校正。短日示例不构成本题年度样本外证据，也不证明复杂预测或控制必然优于简单方法。

## Lago等，2021，Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark

Applied Energy 293, 116983；DOI: https://doi.org/10.1016/j.apenergy.2021.116983 。本地Forecasting_Day_Ahead_Electricity_Prices_2021.pdf共21页，为CC BY期刊正文。第2–9页提出五市场各六年、末两年测试的开放基准，比较LASSO自回归LEAR和DNN；本文ridge不是其LEAR复现，也没有复现该多市场基准。

第10–13页及第17–18页强调简单而强的基线、每日重估、测试与调参分离、公开代码、MAE/rMAE/RMSE及显著性评估，反对只靠短样本和MAPE宣布胜出。这里采用昨日同刻、7日均值、同星期混合、季节中位数和两种ridge，月首仅已完成21日按真实费用选参。rMAE分母明确取昨日基线，区别于文献优先建议的7日前基线。附件仅一年，1月预运行后测试334日，未满足其至少一年测试和多市场建议，因此结论限定于本数据与策略类。费用比较是滚动样本外实测，不是文献意义的普适显著性结论。

## 推导与实施归属

价格加权分位来自本题pG+5p(N-G)+的直接一阶条件；CVaR采用有限分布的标准辅助阈值定义，代码枚举原子精确求该有限分布值。二者不是上述论文已证明适用于本题的结论。储能MPC、联合经验场景和价格更新必须接受同一信息集、同一结算和年度真实执行检验。未实施DRO、GBDT或Conformal，不凭术语增添结论。
''')
if __name__=='__main__':generate()
