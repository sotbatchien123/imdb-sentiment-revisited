"""Atomic output with bounded retries for Windows file viewers/antivirus."""
import json
import os
import time
from pathlib import Path

def atomic_json(path, obj, attempts=10):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    for n in range(attempts):
        try:
            os.replace(tmp,path)
            return True
        except PermissionError:
            if n+1==attempts: raise
            time.sleep(.1)

def publish_status(path,obj):
    """A locked UI status file must never terminate training."""
    for destination in [Path(path),Path(path).with_name('status_live.json')]:
        try:
            atomic_json(destination,obj,attempts=2)
        except OSError as exc:
            print(f'Progress display update deferred for {destination.name}: {exc}',flush=True)
