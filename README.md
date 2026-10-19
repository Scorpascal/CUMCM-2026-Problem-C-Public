# 2026 全国大学生数学建模竞赛 · C 题

**🏆 北京市一等奖** · 微网与外部电网电力调控

从确定性日前调度，到预测误差下的实时执行，再到日内滚动购电与电价波动决策，本项目研究光伏、负荷、储能和外部电网如何协同运行。这里提供经过脱敏的代码、归档论文、五个正式结果工作簿，以及可校验的公开分发包。

奖项由参赛者确认。项目原始开发历史和冻结 ZIP 保留在私有档案中；本仓库使用独立历史，公开包具有新的名称和 SHA-256，不等同于原始冻结包。未公开队员身份或证书。

**快速入口：** [归档论文](paper/C题最终论文.pdf) · [五个结果表](results/) · [Python 代码](code/) · [Q3/Q4 Release](https://github.com/Scorpascal/CUMCM-2026-Problem-C-Public/releases/tag/public-v1.0.0) · [公开验证记录](docs/VALIDATION.md)

## 模型与结果

统一采用十分钟时间粒度，储能容量 12,000 kWh，允许电量范围 1,200–10,800 kWh，最大充放电功率 5,000 kW。

| 问题 | 方法与任务 | 归档总费用 | 结果 |
|---|---|---:|---|
| Q1 | 确定性日前优化，核对该算例的 LP/MILP 一致性 | 35,126.95 元 | [result1.xlsx](results/result1.xlsx) |
| Q2 | 历史预测、月度选参、日前计划与实时执行 | 14,758,543.05 元 | [result2.xlsx](results/result2.xlsx) |
| Q3 | 在 0/6/12/18 时滚动更新购电，逐事件核算结算 | 14,392,539.81 元 | [result3.xlsx](results/result3.xlsx) |
| Q4-2 | 波动电价下的因果预测与固定日前计划 | 15,399,318.84 元 | [result4-2.xlsx](results/result4-2.xlsx) |
| Q4-3 | 波动电价下的因果预测与日内滚动调整 | 15,008,011.51 元 | [result4-3.xlsx](results/result4-3.xlsx) |

Q1 为单日算例；Q2–Q4 的统计区间为 2025-02-01 至 2025-12-31，共 334 天、48,096 个时段。不同问题的信息集、调整权限和储能轨迹不同，费用差额不能直接解释为单一预测模块的净收益。

## 阅读与目录

1. 从[归档论文](paper/C题最终论文.pdf)了解问题、假设、方法与结论。
2. 阅读 [Q1](code/Q1/Q1_final.md)、[Q2](code/Q2/Q2_final.md)、[Q3](code/Q3/Q3_final.md)、[Q4](code/Q4/Q4_final.md) 模型记录，结合 `results/` 和 `intermediate/` 查看结果。
3. 下载公开包检查完整的数值明细；需要重算时，按下文在独立副本中补齐官方输入。

| 路径 | 内容 |
|---|---|
| `code/` | 基于正式提交支撑材料整理的 Python 源码与模型记录 |
| `results/` | 五个归档结果工作簿，重算不会覆盖这一目录 |
| `intermediate/` | 月度选择、敏感性及指定日期等精选结果 |
| `figures/` | 归档图表 |
| `paper/` | 归档论文 PDF 与 AI 工具使用说明 |
| `config/requirements.txt` | NumPy、SciPy、openpyxl、Matplotlib 的公开依赖范围 |
| `docs/` | 输入说明、发布边界、验证记录与来源清单 |
| `tools/verify_public.py` | 仅用 Python 标准库校验公开文件清单 |

模型记录、论文及其中的测试叙述保留归档时的语境。原件冻结状态、实际参赛提交记录、项目获奖信息与本轮公开验证分别记录，不互相替代。

## 下载 Q3 / Q4 公开包

从 [public-v1.0.0 Release](https://github.com/Scorpascal/CUMCM-2026-Problem-C-Public/releases/tag/public-v1.0.0) 下载：

- `CUMCM2026-C-Q3-public-v1.zip`：Q1–Q3 源码、结果及数值实验输出。
- `CUMCM2026-C-Q4-public-v1.zip`：Q1–Q4 源码、结果及数值实验输出，包含 Q4 依赖的前序计算结果。
- 对应 `.zip.sha256`、`SHA256SUMS` 与 `RELEASE_MANIFEST.json`。

将 ZIP 和对应校验文件放在同一目录，在 macOS 执行：

```bash
shasum -a 256 -c CUMCM2026-C-Q3-public-v1.zip.sha256
shasum -a 256 -c CUMCM2026-C-Q4-public-v1.zip.sha256
```

Linux 可用 `sha256sum -c`。两个包均为独立公开衍生版本，不包含旧冻结 ZIP、个人日志、历史 Git 对象或第三方论文全文。具体大小与来源对应关系见 [Release 清单](docs/RELEASE_ASSETS.json)。

解压后，先在包根目录执行文件级检查：

```bash
python3 tools/verify_public.py
```

校验针对 `PUBLIC_MANIFEST.json`，无需安装数值依赖。它验证分发文件的完整性，不替代数值复现，也不声称通过私有原件的冻结校验。重新计算会改变 `code/Q*/` 下的文件，因此应先校验，再在独立工作副本中重算。

## 环境与复现

兼容要求为 Python 3.11 或以上；本轮实际运行环境与验证范围见 [VALIDATION.md](docs/VALIDATION.md)。不需要商业求解器、Node、私有预览工具或在线模型服务。

在克隆仓库或解压后的**独立工作副本**根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r config/requirements.txt
```

请自行从合法官方渠道取得输入，按 [INPUT_DATA.md](docs/INPUT_DATA.md) 放置附件 1–4 和附件 5 的五个空白模板。本仓库及 Release 均不分发这些原始附件。

```bash
# 从 Q1 顺序重算到 Q3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/reproduce.py --through Q3

# 从 Q1 顺序重算到 Q4（完整实验与诊断较耗时）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/reproduce.py --through Q4
```

Git 仓库适合阅读源码与精选结果；Release 额外提供数值实验数组、逐时段表与诊断输出。Q3 包只支持到 Q3，Q4 包和完整仓库支持到 Q4。重算输出写入 `code/Q1`–`code/Q4`，不覆盖 `results/` 中的归档工作簿。

跨平台比较应依据费用、物理约束、信息集、账本与单元格内容；求解器平局解、图表元数据、压缩时间戳可能造成文件散列差异。原工程冻结清单未随公开包分发，其缺席明确记录为 `not_packaged`。

## 口径与边界

- Q1 的日初、日末储能量相等约束不自动延伸至 Q2–Q4。
- 普通购电计划、实际执行与紧急购电分开核算；主模型中的紧急购电按五倍价格结算。
- Q3/Q4-3 仅在允许时点修改尚未执行的承诺，利用逐事件账本复算费用。
- 预测价格、决策时可见信息、事后真实结算价格分开处理；Oracle 仅用于诊断，不是正式策略。
- 时间标签、效率、调减结算与报价可见性的解释属于建模假设，相关替代敏感性保留在模型记录中。

[参考文献](docs/REFERENCES.md)仅保留书目信息。使用与权利说明见 [NOTICE.md](NOTICE.md)，公开副本的来源、排除项和包装改动见 [PUBLICATION.md](docs/PUBLICATION.md)。
