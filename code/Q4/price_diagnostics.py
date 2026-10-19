"""Pre-optimization causal forecast experiment and interval diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'.matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q4_utils import *
from q4_price_forecast import PriceFactory,MODELS

def price_diagnostics():
    writable();dates,a,fixed,p,records=read_inputs();factory=PriceFactory(p)
    outputs={};rows=[];intervals=[]
    baseline=np.array([factory.raw(d,0,'last')[:144] for d in range(365)])
    for model in MODELS:
        pred=np.array([[factory.raw(d,k,model) for k in range(4)] for d in range(365)])
        outputs[model]=pred
        # Same-day targets only: avoid mixing issuance lead times and cross-year missing data.
        for k in range(4):
            t=k*36;n=144-t;y=p[31:,t:];z=pred[31:,k,:n];err=z-y
            masks=[('all','all',np.ones(y.shape,bool))]
            masks += [('month',str(m),np.broadcast_to(np.array([d.month==m for d in dates[31:]])[:,None],y.shape)) for m in range(2,13)]
            masks += [('hour',str(h),np.broadcast_to(np.arange(t,144)[None]//6==h,y.shape)) for h in range(t//6,24)]
            masks += [('lead_hour',str(h),np.broadcast_to(np.arange(n)[None]//6==h,y.shape)) for h in range((n+5)//6)]
            threshold=np.quantile(y,.9)
            masks += [('price_group','top10',y>=threshold),('price_group','bottom10',y<=np.quantile(y,.1))]
            for group,label,mask in masks:
                er=err[mask];yy=y[mask];zz=z[mask];bm=(baseline[31:,t:]-y)[mask]
                if not len(er):continue
                rows.append(dict(model=model,stage=k,group=group,label=label,n=len(er),mae=float(np.abs(er).mean()),rmse=float(np.sqrt(np.mean(er**2))),
                    bias=float(er.mean()),rmae=float(np.abs(er).mean()/max(np.abs(bm).mean(),1e-12)),
                    spike_miss_rate=float(np.mean(zz[yy>=threshold]<threshold)) if np.any(yy>=threshold) else '',
                    diagnostic_weighted_mae=float(np.average(np.abs(er),weights=np.maximum(a[31:,t:,0][mask],1)))))
        coverage={level:[] for level in (.5,.8,.9)}
        for d in range(31,365):
            origins=range(max(1,d-31),d-1)
            r=np.array([p.ravel()[j*144:j*144+288]-pred[j,0] for j in origins])[:,:144]
            for level in coverage:
                q=(1-level)/2;lo=np.maximum(.001,pred[d,0,:144]+np.quantile(r,q,axis=0));hi=np.maximum(.001,pred[d,0,:144]+np.quantile(r,1-q,axis=0))
                residual=p[d]-lo;pinlo=np.maximum(q*residual,(q-1)*residual)
                residual=p[d]-hi;pinhi=np.maximum((1-q)*residual,-q*residual)
                coverage[level].append([np.mean((p[d]>=lo)&(p[d]<=hi)),np.mean(hi-lo),np.mean((pinlo+pinhi)/2)])
        for level,vals in coverage.items():
            v=np.mean(vals,0);intervals.append(dict(model=model,nominal=level,coverage=v[0],width=v[1],pinball=v[2],method='rolling complete historical residual; no coverage guarantee'))
        print('forecast diagnostics',model,flush=True)
    np.savez_compressed(OUT/'experiments/price_forecast_arrays.npz',**outputs)
    write_csv(OUT/'price_metrics.csv',rows);write_csv(OUT/'price_interval_metrics.csv',intervals)
    (OUT/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax_,date_ in zip(ax.ravel(),('2025-03-20','2025-06-21','2025-09-23','2025-12-21')):
        d=next(i for i,v in enumerate(dates) if str(v)==date_);x=np.arange(144)/6
        ax_.plot(x,p[d],label='Realized',color='black',lw=1)
        for model in ('last','mean7','ridge30'):ax_.plot(x,outputs[model][d,0,:144],label=model,lw=1)
        rr=np.array([p[j]-outputs['mean7'][j,0,:144] for j in range(d-30,d)])
        ax_.fill_between(x,np.maximum(.001,outputs['mean7'][d,0,:144]+np.quantile(rr,.05,axis=0)),outputs['mean7'][d,0,:144]+np.quantile(rr,.95,axis=0),alpha=.12)
        ax_.set(title=date_,xlabel='Hour',ylabel='CNY/kWh');ax_.legend(fontsize=8,ncol=2)
    fig.savefig(OUT/'figures/price_forecast_precheck.png',dpi=150);fig.savefig(OUT/'figures/price_forecast_precheck.svg');plt.close(fig)

if __name__=='__main__':price_diagnostics()
