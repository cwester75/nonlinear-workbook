"""Common output helpers for reproducible headless Chapter 1 demonstrations."""
import json
import platform
from pathlib import Path
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20261011

def clean(value):
    if is_dataclass(value): return clean(asdict(value))
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,Fraction): return str(value)
    if isinstance(value,complex): return {'real':clean(value.real),'imag':clean(value.imag)}
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value


def setup(stage):
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter01/'+stage)
    directory=Path(parser.parse_args().output); directory.mkdir(parents=True,exist_ok=True)
    return directory,dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,stage=stage)


def save_report(directory, report):
    (directory/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Saved {directory/'report.json'}")


def save_plot(directory,name,fig):
    fig.savefig(directory/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'nonlinear-workbook'})
    plt.close(fig)


def csv(directory,name,data,header):
    np.savetxt(directory/(name+'.csv'),data,delimiter=',',header=header,comments='')
