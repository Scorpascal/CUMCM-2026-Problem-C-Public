"""Q3/Q4可复用的增量调整账本及确定性候选优化；不读取真实曲线。"""
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import hstack,vstack,csr_matrix,coo_matrix
from microgrid_common import Battery
from microgrid_online import balance_matrix,terminal_value

def settlement(old,new,price,rule='A',original=None):
    """A沉没+额外费用；B逐事件调减退p再收.5p；B_net原计划净偏差重估。
    B_net返回相对上次账面净偏差费用的冲销差额，禁止重复收整笔账。
    """
    old,new,p=[np.asarray(v,float) for v in (old,new,price)]
    u=np.maximum(new-old,0);r=np.maximum(old-new,0)
    up=1.5*p*u;penalty=.5*p*r;refund=np.zeros_like(r)
    if rule=='B':refund=p*r
    elif rule=='B_net':
        assert original is not None
        base=np.asarray(original)
        # Signed changes in cumulative financial components; physical U/R remain positive.
        up=1.5*p*(np.maximum(new-base,0)-np.maximum(old-base,0))
        penalty=.5*p*(np.maximum(base-new,0)-np.maximum(base-old,0))
        refund=p*(np.maximum(base-new,0)-np.maximum(base-old,0))
    elif rule!='A':raise ValueError(rule)
    return dict(U=u,R=r,up_cost=up,penalty=penalty,refund=refund,charge=up+penalty-refund)

def adjustment_plan(net,price,initial,old,value,rule='A',original=None,battery=Battery(),integer=False):
    """当前自然日剩余m点已有承诺，时域其余点只是正常价格前瞻。
    min adjustment + future purchase - value E_end; LP后精确消循环。
    """
    net,p,old=[np.asarray(v,float) for v in (net,price,old)]
    n=len(net);m=len(old);b=battery;size=5*n+1+2*m
    assert n>=m>0 and p.shape==net.shape and np.isfinite(net).all() and (p>0).all()
    base=np.asarray(original) if rule=='B_net' else old
    assert base.shape==old.shape
    obj=np.zeros(size);obj[m:n]=p[m:];obj[5*n]=-value
    obj[5*n+1:5*n+1+m]=1.5*p[:m]
    obj[5*n+1+m:]=(.5 if rule=='A' else -.5)*p[:m]
    lo=np.zeros(size);hi=np.full(size,np.inf)
    hi[n:3*n]=b.power*b.dt;lo[4*n:5*n+1]=b.minimum;hi[4*n:5*n+1]=b.maximum
    lo[4*n]=hi[4*n]=np.clip(initial,b.minimum,b.maximum)
    a=hstack([balance_matrix(n,b.eta_charge,b.eta_discharge),csr_matrix((2*n,2*m))],format='csr')
    t=np.arange(m)
    link=coo_matrix((np.r_[np.ones(m),-np.ones(m),np.ones(m)],
        (np.tile(t,3),np.r_[t,5*n+1+t,5*n+1+m+t])),shape=(m,size)).tocsr()
    a=vstack([a,link],format='csr');rhs=np.r_[net,np.zeros(n),base]
    if integer:
        t=np.arange(n);pm=b.power*b.dt
        mutex=coo_matrix((np.r_[np.ones(n),-pm*np.ones(n),np.ones(n),pm*np.ones(n)],
          (np.r_[t,t,n+t,n+t],np.r_[n+t,size+t,2*n+t,size+t])),shape=(2*n,size+n)).tocsr()
        aa=vstack([hstack([a,csr_matrix((len(rhs),n))]),mutex],format='csr')
        res=milp(np.r_[obj,np.zeros(n)],integrality=np.r_[np.zeros(size),np.ones(n)],
          bounds=Bounds(np.r_[lo,np.zeros(n)],np.r_[hi,np.ones(n)]),
          constraints=LinearConstraint(aa,np.r_[rhs,np.full(2*n,-np.inf)],np.r_[rhs,np.zeros(n),np.full(n,pm)]),
          options={'mip_rel_gap':1e-9,'time_limit':120})
    else:res=linprog(obj,A_eq=a,b_eq=rhs,bounds=np.column_stack([lo,hi]),method='highs')
    if not res.success:raise RuntimeError(res.message)
    x=res.x;g=x[:n].copy();c=x[n:2*n].copy();d=x[2*n:3*n].copy();w=x[3*n:4*n].copy();e=x[4*n:5*n+1].copy()
    delta=np.minimum(c,d/(b.eta_charge*b.eta_discharge))
    c-=delta;d-=b.eta_charge*b.eta_discharge*delta;w+=(1-b.eta_charge*b.eta_discharge)*delta
    assert np.max(np.abs(g+d-c-w-net))<1e-5
    assert np.max(np.abs(np.diff(e)-b.eta_charge*c+d/b.eta_discharge))<1e-5
    return dict(G=g,C=c,D=d,W=w,E=e,objective=float(res.fun))
