"""Q4 local paths, immutable input protection and serialization."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
for p in (ROOT,ROOT/'Q2',ROOT/'Q3'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def check_protected():
    snapshot=OUT/'pre_q4_workspace_hashes.json'
    if not snapshot.exists():
        return 0  # Submission copy omits freeze-only protection metadata.
    hashes=json.loads(snapshot.read_text())
    changes=[p for p,h in hashes.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
    assert not changes,changes
    return len(hashes)

def writable():
    if (OUT/'freeze_manifest.json').exists():raise RuntimeError('Q4 frozen: use an independent reproduction copy')
    check_protected()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]) if rows else (fields or ['empty']))
        w.writeheader();w.writerows(rows)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def read_inputs():
    from microgrid_online import read_year
    from microgrid_forecasts import read_pv_forecasts
    from microgrid_common import end_minutes
    from openpyxl import load_workbook
    dates,a,fixed=read_year(ROOT)
    wb=load_workbook(ROOT/'附件/附件4.xlsx',read_only=True,data_only=True)
    rows=list(wb.active.values);wb.close()
    assert [end_minutes(x) for x in rows[0][1:]]==list(range(10,1441,10))
    assert [r[0].date() for r in rows[1:]]==dates
    realized=np.array([r[1:] for r in rows[1:]],float)
    assert realized.shape==(365,144) and np.isfinite(realized).all() and (realized>0).all()
    return dates,a,fixed,realized,read_pv_forecasts(ROOT/'附件/附件3.xlsx')
