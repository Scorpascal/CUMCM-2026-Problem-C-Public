"""Q2–Q4可复用：全年数据、自然日标签、纯物理计划与实时执行。

不改动Q1公共内核；全部流量为交流母线侧kWh，energy为内部kWh。
本模块不读取未来数据，预测/价格信息由调用者显式传入。
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
from openpyxl import load_workbook
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix
from microgrid_common import Battery, end_minutes, read_single_day

ROOT = Path(__file__).resolve().parent
TOL = 1e-7
PERIODS = [f'{t//6}:{t%6*10:02d}-{(t+1)//6}:{(t+1)%6*10:02d}' for t in range(144)]


def read_year(root=ROOT):
    wb = load_workbook(root/'附件/附件2.xlsx', read_only=True, data_only=True)
    matrices, dates = [], None
    for name in ('小区负载', '光伏发电实际功率'):
        rows = list(wb[name].values)
        assert [end_minutes(v) for v in rows[0][1:]] == list(range(10, 1441, 10))
        ds = [r[0].date() if isinstance(r[0], datetime) else r[0] for r in rows[1:]]
        assert ds == [date(2025, 1, 1)+timedelta(days=i) for i in range(365)]
        if dates is not None:
            assert ds == dates
        dates = ds
        a = np.asarray([r[1:] for r in rows[1:]], float)/6
        assert a.shape == (365, 144) and np.isfinite(a).all() and (a >= 0).all()
        matrices.append(a)
    wb.close()
    return dates, np.stack(matrices, axis=-1), read_single_day(root/'附件/附件1.xlsx')['price']


@lru_cache(maxsize=12)
def balance_matrix(n, ec, ed):
    t = np.arange(n)
    row = np.concatenate([t]*4+[n+t]*4)
    col = np.concatenate([t, n+t, 2*n+t, 3*n+t, 4*n+t+1, 4*n+t, n+t, 2*n+t])
    val = np.concatenate([np.ones(n), -np.ones(n), np.ones(n), -np.ones(n),
                          np.ones(n), -np.ones(n), -ec*np.ones(n), np.ones(n)/ed])
    return coo_matrix((val, (row, col)), shape=(2*n, 5*n+1)).tocsr()


def terminal_value(price, battery=Battery()):
    return float(np.sort(price)[:min(24, len(price))].mean()/battery.eta_charge)


def plan_net(net, price, initial, terminal='cycle', battery=Battery(), integer=False):
    """净负荷计划LP，消循环精确恢复互斥；MILP仅作独立最优性对照。

    terminal='cycle'/'value'/None/数值；net已为kWh。允许计划剩余电量W。
    """
    tic = time.perf_counter()
    net, price = np.asarray(net, float), np.asarray(price, float)
    n = len(net); b = battery; pmax = b.power*b.dt
    assert price.shape == net.shape and n > 0
    assert np.isfinite(net).all() and np.isfinite(price).all() and (price >= 0).all()
    assert b.minimum-1e-6 <= initial <= b.maximum+1e-6
    initial = float(np.clip(initial, b.minimum, b.maximum))
    objective = np.zeros(5*n+1); objective[:n] = price
    lo = np.zeros(5*n+1); hi = np.full(5*n+1, np.inf)
    hi[n:3*n] = pmax; lo[4*n:] = b.minimum; hi[4*n:] = b.maximum
    lo[4*n] = hi[4*n] = initial
    if terminal == 'cycle':
        lo[-1] = hi[-1] = initial
    elif terminal == 'value':
        objective[-1] = -terminal_value(price[:144], b)
    elif terminal is not None:
        lo[-1] = hi[-1] = float(terminal)
    a = balance_matrix(n, b.eta_charge, b.eta_discharge)
    rhs = np.r_[net, np.zeros(n)]
    if integer:
        eq = hstack([a, csr_matrix((2*n, n))], format='csr')
        t = np.arange(n)
        mutex = coo_matrix((np.r_[np.ones(n), -pmax*np.ones(n), np.ones(n), pmax*np.ones(n)],
                             (np.r_[t,t,n+t,n+t], np.r_[n+t,5*n+1+t,2*n+t,5*n+1+t])),
                            shape=(2*n,6*n+1)).tocsr()
        res = milp(np.r_[objective,np.zeros(n)], integrality=np.r_[np.zeros(5*n+1),np.ones(n)],
                   bounds=Bounds(np.r_[lo,np.zeros(n)],np.r_[hi,np.ones(n)]),
                   constraints=LinearConstraint(vstack([eq,mutex],format='csr'),
                     np.r_[rhs,np.full(2*n,-np.inf)],np.r_[rhs,np.zeros(n),np.full(n,pmax)]),
                   options={'mip_rel_gap':1e-9, 'time_limit':120})
    else:
        res = linprog(objective, A_eq=a, b_eq=rhs, bounds=np.column_stack([lo,hi]), method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x[:5*n+1]
    g,c,d,w,e = [v.copy() for v in (x[:n],x[n:2*n],x[2*n:3*n],x[3*n:4*n],x[4*n:])]
    delta = np.minimum(c, d/(b.eta_charge*b.eta_discharge))
    c -= delta; d -= b.eta_charge*b.eta_discharge*delta
    w += (1-b.eta_charge*b.eta_discharge)*delta
    assert np.max(np.abs(g+d-c-w-net)) < 1e-5
    assert np.max(np.abs(np.diff(e)-b.eta_charge*c+d/b.eta_discharge)) < 1e-5
    assert not np.any((c>1e-6)&(d>1e-6))
    return dict(G=g,C=c,D=d,W=w,E=e,objective=float(res.fun),seconds=time.perf_counter()-tic,
                variables=6*n+1 if integer else 5*n+1,constraints=4*n if integer else 2*n,
                cycles_removed=int((delta>1e-7).sum()),mip_gap=float(getattr(res,'mip_gap',0)))


def feedback_step(g, load, pv, energy, battery=Battery()):
    """只接收当前观测和当前E；无法读取未来曲线。"""
    b = battery; limit = b.power*b.dt; r = g+pv-load
    c = min(max(r,0.), limit, max(0., (b.maximum-energy)/b.eta_charge))
    d = min(max(-r,0.), limit, max(0., (energy-b.minimum)*b.eta_discharge))
    h = max(0., -r-d); w = max(0., r-c)
    return c,d,h,w,energy+b.eta_charge*c-d/b.eta_discharge


def execute_day(g, actual, initial, battery=Battery(), fixed=None, idle=False):
    """actual仅供环境逐点揭示；控制器feedback_step不能看到数组。

    fixed=(C,D)为储能完全固定替代口径；idle用于共同启动/无储能。
    """
    b=battery; n=len(g); e=np.empty(n+1);e[0]=initial
    c=np.zeros(n);d=c.copy();h=c.copy();w=c.copy()
    for t,(load,pv) in enumerate(actual):
        if fixed is None and not idle:
            c[t],d[t],h[t],w[t],e[t+1]=feedback_step(g[t],load,pv,e[t],b)
        else:
            if fixed is not None:
                c[t],d[t]=fixed[0][t],fixed[1][t]
            r=g[t]+pv+d[t]-load-c[t]
            h[t]=max(-r,0.); w[t]=max(r,0.)
            e[t+1]=e[t]+b.eta_charge*c[t]-d[t]/b.eta_discharge
    # source allocation: solar first; only remaining solar is curtailed
    unused_grid=np.minimum(w,g)
    pv_curtail=w-unused_grid
    result=dict(G=np.asarray(g).copy(),C=c,D=d,H=h,W=w,E=e,
                unused_grid=unused_grid,pv_curtail=pv_curtail,
                pv_curtail_upper=np.minimum(w,actual[:,1]))
    validate_execution(actual,result,b)
    return result


def validate_execution(actual,s,battery=Battery()):
    b=battery;g,c,d,h,w,e=[s[k] for k in ('G','C','D','H','W','E')]
    err=dict(balance=float(np.max(np.abs(g+actual[:,1]+d+h-actual[:,0]-c-w))),
             state=float(np.max(np.abs(np.diff(e)-b.eta_charge*c+d/b.eta_discharge))),
             bounds=float(max(0,b.minimum-e.min(),e.max()-b.maximum)),
             power=float(max(0,c.max()-b.power*b.dt,d.max()-b.power*b.dt)),
             nonnegative=float(max(0,-min(g.min(),c.min(),d.min(),h.min(),w.min()))),
             simultaneous=int(np.sum((c>1e-6)&(d>1e-6))))
    assert max(err.values())<1e-5,err
    assert np.max(np.abs(h*w))<1e-5
    return err


def merge_emergency(dates, emergency, tol=TOL):
    out=[]
    for date_,day in zip(dates, emergency):
        start=None
        for t in range(len(day)+1):
            positive=t<len(day) and day[t]>tol
            if positive and start is None:
                start=t
            if not positive and start is not None:
                out.append(dict(date=date_.isoformat(),start=start,stop=t,
                    period=f'{start//6}:{start%6*10:02d}-{t//6}:{t%6*10:02d}',
                    kwh=float(day[start:t].sum())))
                start=None
    return out
