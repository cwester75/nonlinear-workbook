"""Optional shadow experiment; no imports into outlierhunter's decision layers."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
from nonlinear_workbook.diagnostic_adapter import WindowConfig,diagnose_prefix


def verify_installation(pin_path):
    pin=json.loads(Path(pin_path).read_text())
    distribution=importlib.metadata.distribution('nonlinear-workbook')
    direct=json.loads(distribution.read_text('direct_url.json') or '{}')
    actual=direct.get('vcs_info',{}).get('commit_id')
    if actual!=pin['workbook_commit']: raise RuntimeError('installed workbook is not the pinned VCS commit; editable checkouts are not accepted')
    if np.__version__!=pin['numpy'] or platform.python_version()!=pin['python']:
        raise RuntimeError('runtime differs from pinned numerical environment')
    return pin


def observe_prefix(values,*,end_index,config,source_id,pin_path):
    pin=verify_installation(pin_path)
    result=diagnose_prefix(values,end_index=end_index,config=config,source_id=source_id,workbook_commit=pin['workbook_commit'])
    result['provenance']['commit_verification']='installed VCS direct_url.json matched pin'
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pin',type=Path,default=Path(__file__).with_name('workbook-pin.json'))
    parser.add_argument('--synthetic-smoke',action='store_true',required=True)
    args=parser.parse_args(); config=WindowConfig(window=256,sample_interval=.1,time_unit='second',value_unit='normalized amplitude',epsilon=.3)
    values=np.sin(np.arange(512)*.1)
    kw=dict(end_index=256,config=config,source_id='synthetic-sine-only',pin_path=args.pin)
    before=observe_prefix(values,**kw); values[256:]=np.nan
    assert before==observe_prefix(values,**kw)
    assert observe_prefix(np.ones(256),**kw)['status']=='partial'
    assert observe_prefix(values,**dict(kw,end_index=32))['status']=='insufficient_window'
    print(json.dumps(dict(status='shadow adapter smoke passed',provenance=before['provenance'],features=before['features']),indent=2,allow_nan=False))


if __name__=='__main__': main()
