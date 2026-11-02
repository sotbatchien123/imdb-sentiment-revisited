"""Read-only training observer producing one status JSON per model."""
import argparse
import ctypes
import datetime
import json
import os
import time
from pathlib import Path
from runtime_io import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full_imdb_status'

def read(path,default=None):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError): return default

def alive(pid):
    if not pid: return None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle: return False if ctypes.get_last_error()==87 else None
    code=ctypes.c_ulong()
    try:
        return code.value==259 if kernel.GetExitCodeProcess(handle,ctypes.byref(code)) else None
    finally: kernel.CloseHandle(handle)

def publish():
    cfg=read(ROOT/'configs/full_imdb.json'); run=ROOT/cfg['output']
    live=read(run/'status_live.json',{})
    running=alive(live.get('pid'))
    now=datetime.datetime.now().astimezone().isoformat()
    files=[]
    for model in cfg['models']:
        seeds=[]
        for seed in cfg['seeds']:
            path=run/f'{model}_seed{seed}'
            history=read(path/'history.json',[]); result=read(path/'test_metrics.json')
            current=live.get('model')==model and live.get('seed')==seed
            if result: state='completed'
            elif current and running is False: state='interrupted'
            elif current: state=live.get('stage','unknown')
            elif (path/'latest.pt').exists(): state='checkpoint_available'
            else: state='queued' if running is True else 'not_started'
            record={'seed':seed,'status':state,'completed_epochs':len(history),
                    'epoch':live.get('epoch') if current and not result else None,
                    'max_epochs':cfg['max_epochs'],
                    'batch':live.get('batch') if current and not result else None,
                    'total_batches':live.get('total_batches') if current and not result else None,
                    'best_validation_macro_f1':max((h['validation']['macro_f1'] for h in history),default=None),
                    'latest_train_loss':live.get('train_loss') if current and live.get('stage')=='training' and not result else (history[-1]['train_loss'] if history else None),
                    'latest_checkpoint':str(path/'latest.pt') if (path/'latest.pt').exists() else None,
                    'best_checkpoint':str(path/'best.pt') if (path/'best.pt').exists() else None,
                    'history_file':str(path/'history.json') if history else None,
                    'test_metrics_file':str(path/'test_metrics.json') if result else None,
                    'test_metrics':result}
            seeds.append(record)
        completed=sum(s['status']=='completed' for s in seeds)
        active=next((s for s in seeds if s['status'] in ['training','validation','final_test','interrupted','paused','failed']),None)
        state='completed' if completed==len(seeds) else (active['status'] if active else ('partially_completed' if completed else ('queued' if running else 'not_started')))
        payload={'model':model,'status':state,'updated_at':now,'source_updated_at':live.get('updated_at'),
                 'training_pid':live.get('pid'),'training_process_alive':running,
                 'completed_runs':completed,'total_runs':len(seeds),
                 'metrics_unit':'fraction_0_1','max_epochs_per_run':cfg['max_epochs'],
                 'early_stopping':{'enabled':cfg.get('early_stopping',True),'minimum_epochs':cfg['min_epochs'],'patience':cfg['patience'],'metric':'validation_macro_f1'},
                 'seeds':seeds,'source_status':str(run/'status_live.json'),
                 'note':'Validation metrics are not test results. Null means not yet available. Observer does not launch training.'}
        dest=OUT/f'{model}.json'; files.append(str(dest))
        try: atomic_json(dest,payload,attempts=3)
        except OSError as exc: print(f'Deferred {model} status update: {exc}',flush=True)
    try: atomic_json(OUT/'index.json',{'updated_at':now,'files':files,'training_process_alive':running,'source_stage':live.get('stage')})
    except OSError: pass
    return live.get('stage') in ['complete','failed','paused'] or running is False

def main():
    p=argparse.ArgumentParser(); p.add_argument('--watch',action='store_true'); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (OUT/'observer.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1): lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: print('Observer already running.'); return
        started=time.time()
        while True:
            done=publish()
            # A restarted trainer imports libraries before replacing its old
            # paused status; allow startup before treating that old status as final.
            if not args.watch or (done and time.time()-started>=60): break
            time.sleep(5)

if __name__=='__main__': main()
