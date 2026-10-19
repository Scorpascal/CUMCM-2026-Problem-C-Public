"""因果特征工厂：控制器只能收到当前特征，不收到全年数据。"""
from dataclasses import dataclass
from datetime import datetime,timedelta
from functools import lru_cache
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Q2'))
from forecast_q2 import forecast
from microgrid_forecasts import interpolate_pv

@dataclass(frozen=True)
class StageFeatures:
    now:int
    source_issue:int
    prediction:np.ndarray
    residual:np.ndarray
    residual_origins:tuple
    residual_latest_end:int
    correction:np.ndarray

def causal_prediction(history,record,source_anchor,day,k,method,gain):
    """history只包含截至当前阶段开始已结束的十分钟记录。"""
    t=k*36;now=day*144+t
    assert history.shape==(now,2)
    assert record.issue<=datetime(2025,1,1)+timedelta(minutes=10*now)
    complete=history[:day*144].reshape(day,144,2)
    pred=forecast(complete,day,'blend',3).reshape(-1,2) if day else np.zeros((432,2))
    # 日内只用最近最多6小时已实现负载误差，衰减时间固定6小时。
    if t and gain:
        delta=(history[day*144+max(0,t-36):now,0]-pred[max(0,t-36):t,0]).mean()
        pred[t:,0]=np.maximum(pred[t:,0]+gain*delta*np.exp(-np.arange(432-t)/36),0)
    curve=interpolate_pv(record,source_anchor,method)/6
    origin=int((record.issue-datetime(2025,1,1)).total_seconds()/600)-day*144
    lo=max(0,origin);hi=min(432,origin+144)
    pred[lo:hi,1]=curve[lo-origin:hi-origin]
    return pred[t:t+288].copy()

class FeatureFactory:
    """环境拥有数据；只有此工厂按时间切片后调用纯函数。缓存不用于模型选参。"""
    def __init__(self,actual,records):
        self.actual=np.asarray(actual);self.flat=self.actual.reshape(-1,2)
        self.records=records

    @lru_cache(maxsize=8192)
    def raw(self,day,k,method,gain,source):
        now=day*144+k*36;issue=day*144+source*36
        dt=datetime(2025,1,1)+timedelta(minutes=issue*10)
        rec=self.records[dt]
        anchor=self.flat[issue-1,1]*6 if issue else 0.
        return causal_prediction(self.flat[:now],rec,anchor,day,k,method,gain)

    @lru_cache(maxsize=768)
    def feature(self,day,k,method='linear',gain=0.,bias=False,source=None):
        source=k if source is None else source
        assert source<=k
        now=day*144+k*36
        pred=self.raw(day,k,method,gain,source).copy()
        # 完整48小时残差窗口必须全部实现；即使只用自然日剩余部分也不借用未完成尾部。
        origins=tuple(range(max(1,day-31),max(1,day-1)))
        residual=[]
        for d in origins:
            start=d*144+k*36;end=start+288
            assert end<=now
            residual.append(self.flat[start:end]-self.raw(d,k,method,gain,source))
        r=np.array(residual).reshape(-1,288,2)
        correction=np.zeros(288)
        if bias and len(r):
            correction=r[:,:,1].mean(axis=0)
            adjusted=np.maximum(0,pred[:,1]+correction)
            correction=adjusted-pred[:,1];pred[:,1]=adjusted;r[:,:,1]-=correction
        latest=(origins[-1]*144+k*36+288) if origins else -1
        return StageFeatures(now,day*144+source*36,pred,r,origins,latest,correction)

def risk_net(f,q,n):
    margin=np.zeros(n)
    if q and len(f.residual):margin=np.maximum(0,np.quantile(f.residual[:,:n,0]-f.residual[:,:n,1],q,axis=0))
    return f.prediction[:n,0]-f.prediction[:n,1]+margin,margin

def scenarios(f,n,count=8):
    r=f.residual
    if not len(r):return f.prediction[None,:n].copy()
    indices=np.unique(np.linspace(0,len(r)-1,min(count,len(r))).astype(int))
    return np.maximum(0,f.prediction[None,:n]+r[indices,:n])
