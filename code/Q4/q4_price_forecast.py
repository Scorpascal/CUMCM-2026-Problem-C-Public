"""Price forecasts issued from a strict historical prefix, never a future realized vector."""
from functools import lru_cache
import numpy as np

MODELS=('last','mean7','blend','seasonal','ridge30','ridge_expanding')

def daily_price(history,day,model):
    history=np.asarray(history,float)
    assert history.shape==(day,144)
    if day==0:return np.full((3,144),.5)
    last=history[-1];recent=history[-7:].mean(0)
    weekly=[]
    for j in range(3):
        ix=[d for d in range(day) if (day+j-d)%7==0][-4:]
        weekly.append(history[ix].mean(0) if ix else recent)
    if model=='last':out=np.tile(last,(3,1))
    elif model=='mean7':out=np.tile(recent,(3,1))
    elif model=='blend':out=.5*recent+.5*np.array(weekly)
    elif model=='seasonal':
        # Calendar season inferred only from observed days, 90-day rolling median.
        out=[]
        for j in range(3):
            ix=[d for d in range(max(0,day-90),day) if ((d+2)%7<5)==((day+j+2)%7<5)]
            out.append(np.median(history[ix],0) if ix else recent)
        out=np.array(out)
    elif model in ('ridge30','ridge_expanding'):
        if day<9:return .5*recent+.5*np.array(weekly)
        # Pooled same-slot autoregression; training targets all strictly before issue day.
        def design(h,k):
            x=np.arange(144)/144*2*np.pi
            return np.column_stack([np.ones(144),h[k-1],h[max(0,k-7)],h[max(0,k-7):k].mean(0),
                np.sin(x),np.cos(x),np.sin(2*x),np.cos(2*x),np.full(144,((k+2)%7)>=5)])
        begin=max(7,day-30) if model=='ridge30' else 7
        xs=np.concatenate([design(history,k) for k in range(begin,day)])
        ys=history[begin:day].ravel()
        reg=np.eye(xs.shape[1])*10.;reg[0,0]=1e-8
        coef=np.linalg.solve(xs.T@xs+reg,xs.T@ys)
        h=history.copy();out=[]
        for j in range(3):
            y=np.maximum(.001,design(h,day+j)@coef);out.append(y);h=np.vstack([h,y])
        out=np.array(out)
    else:raise ValueError(model)
    assert out.shape==(3,144) and np.isfinite(out).all()
    return np.maximum(.001,out)

class PriceFactory:
    def __init__(self,realized):
        self.realized=np.asarray(realized,float)
        self.flat=self.realized.ravel()

    @lru_cache(maxsize=2400)
    def base(self,day,model):return daily_price(self.realized[:day],day,model)

    @lru_cache(maxsize=12000)
    def raw(self,day,k,model):
        t=k*36;now=day*144+t
        base=self.base(day,model).ravel()
        pred=base[t:t+288].copy()
        if t:
            # Stage price update: last hour deviation; two-hour exponential decay.
            delta=np.mean(self.flat[now-6:now]-base[t-6:t])
            pred+=delta*np.exp(-np.arange(288)/12)
        return np.maximum(.001,pred)

    def execution_curve(self,day,t,model,stage_curve,stage_start,n):
        # Only prices ending strictly before t enter future forecast correction.
        now=day*144+t;j=t-stage_start
        p=stage_curve[j:j+n].copy()
        if j:
            count=min(6,j)
            delta=np.mean(self.flat[now-count:now]-stage_curve[j-count:j])
            p+=delta*np.exp(-np.arange(n)/12)
        return np.maximum(.001,p)
