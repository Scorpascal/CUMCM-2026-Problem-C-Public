"""逐日起点、多个时距预测。只接收已经完成日；所有数值为kWh。"""
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import numpy as np

METHODS=('last','mean7','blend')


def forecast(history, day, method, horizon=3):
    """history包含恰好day个已完成日；输出horizon×144×(load,pv)。"""
    history=np.asarray(history,float)
    assert len(history)==day and day>0
    recent=history[-7:].mean(axis=0)
    out=[]
    for j in range(horizon):
        if method=='last':
            y=history[-1]
        elif method=='mean7':
            y=recent
        elif method=='blend':
            inds=[k for k in range(day) if (day+j-k)%7==0][-4:]
            weekly=history[inds].mean(axis=0) if inds else recent
            y=.5*recent+.5*weekly
        else:
            raise ValueError(method)
        out.append(np.maximum(y,0))
    return np.asarray(out)


@dataclass(frozen=True)
class Features:
    day:int
    prediction:dict
    residual:dict
    residual_dates:dict
    training_max:int
    history_sha256:str


class OnlineForecast:
    def __init__(self):
        self.history=[]
        self.issued=[]

    def issue(self):
        day=len(self.history)
        assert len(self.issued)==day
        pred={m:forecast(self.history,day,m) if day else np.zeros((3,144,2)) for m in METHODS}
        residual={}; residual_dates={}
        for method in METHODS:
            residual[method]=[];residual_dates[method]=[]
            for j in range(3):
                # origin>=1 excludes intentionally non-forecast startup day
                origins=list(range(max(1,day-j-30),day-j))
                r=[self.history[k+j]-self.issued[k].prediction[method][j] for k in origins]
                residual[method].append(np.asarray(r).reshape(-1,144,2))
                residual_dates[method].append([k+j for k in origins])
        a=np.asarray(self.history,dtype=float)
        features=Features(day,pred,residual,residual_dates,day-1,hashlib.sha256(a.tobytes()).hexdigest())
        self.issued.append(features)
        return features

    def observe(self,actual):
        assert len(self.issued)==len(self.history)+1
        a=np.asarray(actual,float).copy()
        assert a.shape==(144,2) and np.isfinite(a).all() and (a>=0).all()
        self.history.append(a)


def build_features(actual):
    bank=OnlineForecast()
    for day in actual:
        bank.issue()
        bank.observe(day)
    return bank.issued


def risk_net(features,method,q,horizon):
    pred=features.prediction[method][:horizon]
    net=pred[:,:,0]-pred[:,:,1]
    margin=np.zeros_like(net)
    if q>0:
        for j in range(horizon):
            r=features.residual[method][j]
            if len(r):
                margin[j]=np.maximum(0,np.quantile(r[:,:,0]-r[:,:,1],q,axis=0))
    return (net+margin).ravel(),margin
