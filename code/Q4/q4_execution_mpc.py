"""Fixed-commitment storage MPC. H never buys extra energy for charging."""
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from microgrid_online import feedback_step

@lru_cache(maxsize=144)
def state_matrix(n):
    t=np.arange(n)
    return coo_matrix((np.r_[-.9*np.ones(n),np.ones(n)/.9,-np.ones(n),np.ones(n)],
        (np.tile(t,4),np.r_[t,n+t,2*n+t,2*n+t+1])),shape=(n,3*n+1)).tocsr()

def mpc_step(g,net_forecast,price_forecast,energy,value=0.):
    """Current net/price passed explicitly; tail consists only of legal forecasts.
    For fixed G, charge <= surplus and discharge <= deficit. Min 5pH-vE_end.
    Current-period aggregate demand is an inherited feedback approximation.
    """
    g=np.asarray(g);net=np.asarray(net_forecast);p=np.asarray(price_forecast)
    assert g.shape==net.shape==p.shape and np.isfinite(p).all() and (p>0).all()
    r=g-net;n=len(g)
    # With free disposal and positive stored-energy value, absorbing surplus cannot hurt.
    if r[0]>=0 or energy<=1200+1e-8:
        return feedback_step(g[0],net[0],0.,energy)
    surplus=np.minimum(np.maximum(r,0),5000/6)
    deficit=np.minimum(np.maximum(-r,0),5000/6)
    obj=np.r_[np.zeros(n),-5*p,np.zeros(n),-value]
    lo=np.r_[np.zeros(2*n),np.full(n+1,1200.)];hi=np.r_[surplus,deficit,np.full(n+1,10800.)]
    lo[2*n]=hi[2*n]=float(np.clip(energy,1200,10800))
    sol=linprog(obj,A_eq=state_matrix(n),b_eq=np.zeros(n),bounds=np.column_stack([lo,hi]),method='highs')
    if not sol.success:raise RuntimeError(sol.message)
    c=max(0.,sol.x[0]);d=max(0.,min(sol.x[n],-r[0],5000/6,(energy-1200)*.9))
    h=max(0.,-r[0]-d);w=max(0.,r[0]-c)
    return c,d,h,w,energy+.9*c-d/.9
