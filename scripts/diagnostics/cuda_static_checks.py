"""Giai đoạn 0 (kế hoạch C): kiểm tra tĩnh trên CPU cho lỗi CUDA của multichunk_v1.

Chỉ đọc: không ghi vào work_dirs/multichunk_v1, không dùng GPU, không đọc test.
Kết quả ghi vào reports/chunk_attention_v1/static_checks.json.
"""
import hashlib
import json

import os
import platform
import random
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import transformers
from transformers import AutoTokenizer
from phase2_core import ROOT, chunks_from_ids
from runtime_io import atomic_json

RUN=ROOT/'work_dirs/multichunk_v1'
OUT=ROOT/'reports/chunk_attention_v1/static_checks.json'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def finite_tensors(obj):
    """Đếm tensor và số tensor có giá trị không hữu hạn trong state lồng nhau."""
    total=bad=0
    stack=[obj]
    while stack:
        x=stack.pop()
        if torch.is_tensor(x):
            if x.is_floating_point():
                total+=1; bad+=int(not torch.isfinite(x).all())
        elif isinstance(x,dict): stack.extend(x.values())
        elif isinstance(x,(list,tuple)): stack.extend(x)
    return total,bad

def main():
    cfg=json.loads((ROOT/'configs/multichunk_pilot.json').read_text())
    report={'purpose':'Static CPU checks for multichunk_v1 seed42 CUDA failure; read-only','checks':{}}
    files=['configs/multichunk_pilot.json','scripts/train_multichunk.py','scripts/phase2_core.py','scripts/train_baselines.py',
           f"{cfg['dataset']}/train.jsonl",f"{cfg['dataset']}/validation.jsonl",
           'work_dirs/multichunk_v1/seed42/latest.pt','work_dirs/multichunk_v1/seed42/best.pt',
           'work_dirs/multichunk_v1/seed42/history.json','work_dirs/multichunk_v1/status_live.json']
    report['checksums_sha256']={f:sha256(ROOT/f) for f in files}
    report['environment']={'python':sys.version.split()[0],'platform':platform.platform(),'torch':torch.__version__,
                           'torch_cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'transformers':transformers.__version__}

    # Chữ ký giống hệt runner: nếu khớp thì latest.pt vẫn resume được với mã/config/dữ liệu hiện tại.
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
    for name in ['train_multichunk.py','phase2_core.py']: digest.update((ROOT/'scripts'/name).read_bytes())
    for split in ['train','validation']: digest.update((ROOT/cfg['dataset']/f'{split}.jsonl').read_bytes())
    signature=digest.hexdigest()

    latest=torch.load(RUN/'seed42/latest.pt',map_location='cpu',weights_only=False)
    best=torch.load(RUN/'seed42/best.pt',map_location='cpu',weights_only=False)
    history=json.loads((RUN/'seed42/history.json').read_text())
    head_total,head_bad=finite_tensors(latest['head'])
    opt_total,opt_bad=finite_tensors(latest['optimizer'])
    report['checks']['checkpoint']={
        'signature_current':signature,'signature_latest':latest['signature'],'signature_best':best['signature'],
        'signature_matches':signature==latest['signature']==best['signature'],
        'latest_epoch':latest['epoch'],'best_epoch':best['epoch'],'best_validation_f1':best['validation']['macro_f1'],
        'history_matches_latest':history==latest['history'],'history_epochs':[r['epoch'] for r in history],
        'head_float_tensors':head_total,'head_nonfinite_tensors':head_bad,
        'optimizer_float_tensors':opt_total,'optimizer_nonfinite_tensors':opt_bad,
        'optimizer_step':int(max(s['step'] for s in latest['optimizer']['state'].values())),
    }

    tok=AutoTokenizer.from_pretrained(ROOT/cfg['encoder'],local_files_only=True)
    vocab=tok.vocab_size; data={}; token_report={}
    for split in ['train','validation']:
        rows=[json.loads(l) for l in (ROOT/cfg['dataset']/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        ids=tok([r['text'] for r in rows],truncation=False,verbose=False)['input_ids']
        items=[]; lo=10**9; hi=-1; min_len=10**9; max_len=0; coverage_ok=True
        for row,seq in zip(rows,ids):
            chunks=chunks_from_ids(seq,tok.cls_token_id,tok.sep_token_id)
            coverage_ok&=[t for c in chunks for t in c[1:-1]]==seq[1:-1]
            for c in chunks:
                lo=min(lo,min(c)); hi=max(hi,max(c)); min_len=min(min_len,len(c)); max_len=max(max_len,len(c))
            items.append({'review_id':row['review_id'],'chunks':chunks,'label':row['label']})
        data[split]=items
        token_report[split]={'reviews':len(items),'chunks':sum(len(x['chunks']) for x in items),
            'max_chunks':max(len(x['chunks']) for x in items),'min_token_id':lo,'max_token_id':hi,'vocab_size':vocab,
            'token_ids_in_range':0<=lo and hi<vocab,'min_chunk_len':min_len,'max_chunk_len':max_len,
            'chunk_len_within_256':max_len<=256,'labels':sorted({x['label'] for x in items}),'coverage_ok':coverage_ok}
    report['checks']['tokens']=token_report

    # Tái tạo đúng thứ tự review của epoch 6: runner khôi phục RNG Python từ latest.pt rồi random.shuffle.
    random.setstate(latest['rng']['python'])
    order=list(range(len(data['train']))); random.shuffle(order)
    bs=cfg['batch_size']; window=bs*cfg['gradient_accumulation']; mb=cfg['chunk_microbatch']
    batches=[]
    for ws in range(0,len(order),window):
        end=min(ws+window,len(order))
        for s in range(ws,end,bs):
            idx=order[s:min(s+bs,end)]
            batches.append({'batch':len(batches),'reviews_done_after':min(s+bs,end),'chunks':sum(len(data['train'][i]['chunks']) for i in idx),
                            'review_ids':[data['train'][i]['review_id'] for i in idx]})
    epoch5_train_seconds=history[-1]['seconds']
    elapsed=( (23*3600+42*60+31) - (23*3600+34*60) )  # latest.pt lưu ~23:34, lỗi 23:42:31
    frac=min(1.0,elapsed/epoch5_train_seconds)
    guess=int(frac*len(batches))
    report['checks']['epoch6_order']={
        'batches':len(batches),'microbatch_chunks':mb,'max_chunks_in_batch':max(b['chunks'] for b in batches),
        'first_review_ids':batches[0]['review_ids'],
        'note':'Chỉ ước lượng vị trí lỗi theo thời gian (epoch5 dài hơn do tranh GPU); không xác định được batch chính xác vì status cuối bị ghi đè bởi trạng thái failed.',
        'estimated_failure_fraction_of_epoch':round(frac,3),'estimated_batch_window':[max(0,guess-1000),min(len(batches),guess+1000)],
    }
    atomic_json(RUN.parent.parent/'reports/chunk_attention_v1/epoch6_order.json',batches)
    ok=(report['checks']['checkpoint']['signature_matches'] and head_bad==0 and opt_bad==0
        and all(v['token_ids_in_range'] and v['chunk_len_within_256'] and v['coverage_ok'] for v in token_report.values()))
    report['all_static_checks_passed']=bool(ok)
    atomic_json(OUT,report); print(json.dumps(report,ensure_ascii=True,indent=1)[:4000])

if __name__=='__main__': main()
