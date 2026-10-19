"""Q42 and Q43 feature sets and synchronized historical 48-hour residuals."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from q4_utils import *
from forecast_q2 import forecast
from q3_forecast import FeatureFactory
from q4_price_forecast import PriceFactory

@dataclass(frozen=True)
class Features:
    now:int
    source_issue:int
    prediction:np.ndarray
    price_forecast:np.ndarray
    residual:np.ndarray
    origins:tuple
    latest_end:int

class Factory:
    def __init__(self,actual,realized,records,fixed):
        self.actual=np.asarray(actual);self.flat=self.actual.reshape(-1,2)
        self.prices=PriceFactory(realized);self.pv=FeatureFactory(actual,records)
        self.fixed=fixed;self.records=records

    @lru_cache(maxsize=4000)
    def raw_net(self,kind,day,k,source):
        if kind==43:return self.pv.raw(day,k,'linear',0.,source)
        pred=forecast(self.actual[:day],day,'blend',3).reshape(-1,2) if day else np.zeros((432,2))
        return pred[k*36:k*36+288].copy()

    @lru_cache(maxsize=1024)
    def feature(self,kind,day,k,model,stale_price=False,stale_pv=False):
        now=day*144+k*36;source=0 if stale_pv else k
        pred=self.raw_net(kind,day,k,source).copy()
        def pp(d):
            if model=='fixed':return np.tile(self.fixed,3)[k*36:k*36+288].copy()
            return self.prices.base(d,model).ravel()[k*36:k*36+288].copy() if stale_price else self.prices.raw(d,k,model).copy()
        price=pp(day)
        origins=tuple(range(max(1,day-31),max(1,day-1)))
        rs=[]
        for d in origins:
            start=d*144+k*36;end=start+288
            assert end<=now
            rs.append(np.column_stack([self.flat[start:end]-self.raw_net(kind,d,k,source),self.prices.flat[start:end]-pp(d)]))
        residual=np.asarray(rs).reshape(-1,288,3)
        return Features(now,day*144+source*36 if kind==43 else day*144,pred,price,residual,origins,
                        origins[-1]*144+k*36+288 if origins else -1)
