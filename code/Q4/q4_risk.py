"""Empirical joint scenarios, weighted quantiles and finite-sample CVaR."""
import numpy as np

def weighted_quantile(values,weights,q):
    v,w=np.asarray(values,float),np.asarray(weights,float)
    assert v.shape==w.shape and (w>=0).all() and np.all(w.sum(0)>0)
    idx=np.argsort(v,axis=0,kind='stable');v=np.take_along_axis(v,idx,0);w=np.take_along_axis(w,idx,0)
    c=np.cumsum(w,0);i=np.argmax(c>=q*c[-1],axis=0)
    return v[i,np.arange(v.shape[1])]

def cvar(loss,alpha=.9,weights=None):
    x=np.asarray(loss,float);w=np.ones(len(x))/len(x) if weights is None else np.asarray(weights,float)
    assert 0<=alpha<1 and np.isclose(w.sum(),1) and (w>=0).all()
    # Exact weighted discrete Rockafellar-Uryasev minimum, including fractional tail atom.
    return float(min(z+np.dot(w,np.maximum(x-z,0))/(1-alpha) for z in x))

def joint_paths(prediction,price_forecast,residual,count=8,independent=False):
    n=len(price_forecast)
    if not len(residual):return prediction[None,:n].copy(),price_forecast[None].copy(),np.array([-1])
    ix=np.unique(np.linspace(0,len(residual)-1,min(count,len(residual))).astype(int))
    r=residual[ix,:n].copy()
    if independent and len(ix)>1:r[:,:,2]=np.roll(r[:,:,2],1,axis=0)
    # One row index and target timeline shared by all three variables.
    return np.maximum(0,prediction[None,:n]+r[:,:,:2]),np.maximum(.001,price_forecast[None]+r[:,:,2]),ix
