"""Q3/Q4可复用：显式发行时间的官方光伏预报与因果插值。功率kW。"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import numpy as np
from openpyxl import load_workbook
from scipy.interpolate import PchipInterpolator

@dataclass(frozen=True)
class PVForecast:
    issue: datetime
    targets: tuple
    power: tuple

    def __post_init__(self):
        assert self.issue.hour in (0,6,12,18) and self.issue.minute == 0
        assert self.targets == tuple(self.issue+timedelta(hours=h) for h in range(1,25))
        assert len(self.power)==24 and np.isfinite(self.power).all() and min(self.power)>=0

def read_pv_forecasts(path):
    wb=load_workbook(path,read_only=True,data_only=True)
    rows=list(wb.active.values);wb.close()
    assert rows[0][:2]==('日期','预报时刻')
    leads=[int(re.fullmatch(r'预报(\d+)小时',v).group(1)) for v in rows[0][2:]]
    assert leads==list(range(1,25))
    out={}; date_=None
    for row in rows[1:]:
        if row[0]:
            date_=datetime.strptime(str(row[0]),'%Y-%m-%d') if not isinstance(row[0],datetime) else row[0]
        assert date_ is not None
        h,m=map(int,str(row[1]).split(':')[:2]);issue=date_.replace(hour=h,minute=m)
        assert issue not in out
        out[issue]=PVForecast(issue,tuple(issue+timedelta(hours=j) for j in leads),tuple(float(v) for v in row[2:]))
    expected=[datetime(2025,1,1)+timedelta(hours=6*i) for i in range(365*4)]
    assert list(out)==expected
    return out

def interpolate_pv(record,anchor_kw,method='linear'):
    """144个区间结束点的预报功率；anchor为发行时已经观测到的最后一个点。

    zero-order：每个整点值向后保持，首小时使用已知锚点；整点严格通过原值。
    以结束点采样功率近似时段均值，与Q1/Q2离散化一致；不是精确积分。
    """
    x=np.arange(25,dtype=float); y=np.r_[max(0,anchor_kw),record.power]
    t=np.arange(1,145)/6
    if method=='constant':z=y[np.floor(t+1e-10).astype(int)]
    elif method=='linear':z=np.interp(t,x,y)
    elif method=='pchip':z=PchipInterpolator(x,y)(t)
    else:raise ValueError(method)
    z=np.maximum(z,0)
    assert np.allclose(z[5::6],record.power,rtol=0,atol=1e-8)
    return z

def released_curve(records,now,anchor_kw,method='linear'):
    """接受已发布子集；传入任何未来预报直接拒绝。"""
    assert records and all(r.issue<=now for r in records)
    record=max(records,key=lambda r:r.issue)
    return record,interpolate_pv(record,anchor_kw,method)
