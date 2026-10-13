"""Output helpers for deterministic Chapter 11 examples."""
import argparse
import json
import platform
from pathlib import Path
from dataclasses import asdict, is_dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20261012


def clean(value):
    if is_dataclass(value): return clean(asdict(value))
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float): return value if np.isfinite(value) else None
    return value


def setup(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/chapter11/'+stage)
    out = Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    return out,dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,stage=stage)


def save_report(out,report):
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Saved {out/'report.json'}")


def save_plot(out,name,fig):
    fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'nonlinear-workbook'})
    plt.close(fig)


def csv(out,name,data,header):
    np.savetxt(out/(name+'.csv'),data,delimiter=',',header=header,comments='')
