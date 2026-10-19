"""可复用的十分钟数据读取、储能调度与物理校验。所有流量单位为 kWh。"""
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix, hstack, vstack, csr_matrix


@dataclass(frozen=True)
class Battery:
    capacity: float = 12000.0
    minimum: float = 1200.0
    maximum: float = 10800.0
    power: float = 5000.0
    eta_charge: float = 0.9
    eta_discharge: float = 0.9
    dt: float = 1 / 6


def clock_label(minutes):
    return f"{minutes // 60}:{minutes % 60:02d}"


def end_minutes(value):
    """兼容附件中的 Excel time、字符串时间和次日 0:00。"""
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    text = str(value).strip()
    day = 1440 if '+1' in text else 0
    h, m, *_ = text.replace('+1', '').split(':')
    return day + int(h) * 60 + int(m)


def read_single_day(path: Path):
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = list(wb.active.values)
    wb.close()
    if len(rows) != 145 or list(rows[0]) != ['时间', '电价', '小区负载', '光伏发电预测功率']:
        raise ValueError('附件1应为一个表头和144条十分钟记录，列名需与原附件一致')
    ends = np.array([end_minutes(r[0]) for r in rows[1:]])
    if not np.array_equal(ends, np.arange(10, 1441, 10)):
        raise ValueError('时间缺失、重复或错位；期望0:10至24:00的时段结束时间')
    values = np.array([r[1:4] for r in rows[1:]], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('数据必须是有限非负数')
    return dict(price=values[:, 0], load=values[:, 1] / 6,
                pv=values[:, 2] / 6, end_minutes=ends)


def optimize(price, load, pv, battery=Battery(), initial=6000., terminal=6000., integer=False):
    """纯购电费最小化。负载/光伏传入电量；terminal=None可不固定末状态。

    变量顺序 g,c,d,w,e；g+c等为交流母线侧电量，e为电池内部储电量。
    integer=True增加充放电互斥二进制变量；LP用作全局成本下界。
    此函数仅是确定性调度内核，Q2起的预测、场景和费用逻辑需另外实现。
    """
    price, load, pv = [np.asarray(v, dtype=float) for v in (price, load, pv)]
    n = len(price)
    if not n or load.shape != price.shape or pv.shape != price.shape:
        raise ValueError('价格、负载和光伏维度不一致')
    if not all(np.isfinite(v).all() for v in (price, load, pv)):
        raise ValueError('输入包含非有限值')
    if (price < 0).any() or (load < 0).any() or (pv < 0).any():
        raise ValueError('本内核要求非负电价、负载与光伏')
    b = battery
    if not (0 < b.eta_charge <= 1 and 0 < b.eta_discharge <= 1 and
            0 <= b.minimum <= initial <= b.maximum <= b.capacity and b.dt > 0 and b.power >= 0):
        raise ValueError('储能参数无效')
    if terminal is not None and not b.minimum <= terminal <= b.maximum:
        raise ValueError('末状态越界')
    size = 5*n+1
    objective = np.zeros(size); objective[:n] = price
    lower = np.zeros(size); upper = np.full(size, np.inf)
    upper[n:3*n] = b.power*b.dt
    upper[3*n:4*n] = pv
    lower[4*n:] = b.minimum; upper[4*n:] = b.maximum
    lower[4*n] = upper[4*n] = initial
    if terminal is not None:
        lower[-1] = upper[-1] = terminal
    a = lil_matrix((2*n, size))
    for t in range(n):
        a[t, t] = 1; a[t, n+t] = -1; a[t, 2*n+t] = 1; a[t, 3*n+t] = -1
        a[n+t, 4*n+t+1] = 1; a[n+t, 4*n+t] = -1
        a[n+t, n+t] = -b.eta_charge; a[n+t, 2*n+t] = 1/b.eta_discharge
    rhs = np.r_[load-pv, np.zeros(n)]
    a = a.tocsr()
    if integer:
        eq = hstack([a, csr_matrix((2*n, n))], format='csr')
        mutex = lil_matrix((2*n, size+n))
        for t in range(n):
            mutex[t,n+t] = 1; mutex[t,size+t] = -b.power*b.dt
            mutex[n+t,2*n+t] = 1; mutex[n+t,size+t] = b.power*b.dt
        constraints = LinearConstraint(vstack([eq,mutex.tocsr()],format='csr'),
            np.r_[rhs,np.full(2*n,-np.inf)], np.r_[rhs,np.zeros(n),np.full(n,b.power*b.dt)])
        res = milp(np.r_[objective,np.zeros(n)], integrality=np.r_[np.zeros(size),np.ones(n)],
                   bounds=Bounds(np.r_[lower,np.zeros(n)],np.r_[upper,np.ones(n)]),
                   constraints=constraints, options={'mip_rel_gap':1e-9,'time_limit':120})
    else:
        res = linprog(objective, A_eq=a, b_eq=rhs, bounds=list(zip(lower,upper)), method='highs')
    if not res.success:
        raise RuntimeError(f'优化求解失败: {res.message}')
    x = res.x[:size]
    return dict(grid=x[:n], charge=x[n:2*n], discharge=x[2*n:3*n],
                curtail=x[3*n:4*n], energy=x[4*n:], cost=float(price@x[:n]),
                solver_message=res.message, mip_gap=float(getattr(res,'mip_gap',0.0)))


def validate(data, solution, battery=Battery(), initial=6000., terminal=6000., tol=1e-6):
    """从导出所用的决策量独立重算逐时段物理约束，不依赖求解器状态。"""
    s, b = solution, battery
    g,c,d,w,e = [s[k] for k in ('grid','charge','discharge','curtail','energy')]
    balance = g + data['pv'] - w + d - data['load'] - c
    reconstructed = np.r_[initial, initial + np.cumsum(b.eta_charge*c-d/b.eta_discharge)]
    checks = {
        'balance_max_error_kwh':float(np.max(np.abs(balance))),
        'state_max_error_kwh':float(np.max(np.abs(reconstructed-e))),
        'initial_error_kwh':float(abs(e[0]-initial)),
        'terminal_error_kwh':float(abs(e[-1]-terminal)) if terminal is not None else 0.,
        'nonnegative_violation_kwh':float(max(0.,-min(g.min(),c.min(),d.min(),w.min()))),
        'energy_bound_violation_kwh':float(max(0.,b.minimum-e.min(),e.max()-b.maximum)),
        'power_bound_violation_kw':float(max(0.,max(c.max(),d.max())/b.dt-b.power)),
        'curtail_bound_violation_kwh':float(max(0.,np.max(w-data['pv']))),
        'simultaneous_charge_discharge_count':int(np.sum((c>tol)&(d>tol))),
        'cost_error_yuan':float(abs(data['price']@g-s['cost'])),
    }
    if any(v > tol for v in checks.values()):
        raise AssertionError(checks)
    return checks
