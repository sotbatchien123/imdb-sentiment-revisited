"""Giai đoạn 0 (kế hoạch C): tái hiện có kiểm soát epoch 6 của multichunk_v1/seed42.

- Nạp latest.pt (sau epoch 5) ở chế độ chỉ đọc, khôi phục head/optimizer/RNG đúng như runner,
  rồi chạy lại đúng thứ tự batch của epoch 6. Không lưu checkpoint, không chạy validation, không đọc test.
- Mặc định bật CUDA_LAUNCH_BLOCKING=1 (chỉ cho tiến trình này) để traceback chỉ đúng kernel lỗi.
- Ghi log từng batch vào reports/chunk_attention_v1/replay/ và cập nhật mục "replay" trong cuda_diagnosis.json.
- Từ chối chạy nếu GPU đang có tiến trình compute khác, trừ khi truyền --allow-shared-gpu.

Ví dụ:
  .venv-train\\Scripts\\python.exe -I scripts\\diagnostics\\cuda_replay_epoch6.py --max-batches 4000
  .venv-train\\Scripts\\python.exe -I scripts\\diagnostics\\cuda_replay_epoch6.py --device cpu --max-batches 3   (kiểm tra script)
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
parser.add_argument('--max-batches',type=int,default=None,help='Số batch tối đa của epoch 6 (mặc định: cả epoch)')
parser.add_argument('--no-launch-blocking',action='store_true',help='Tắt CUDA_LAUNCH_BLOCKING để đo tốc độ gần với lượt gốc')
parser.add_argument('--allow-shared-gpu',action='store_true',help='Cho phép chạy khi GPU có tiến trình compute khác')
parser.add_argument('--tag',default=None,help='Tên lượt; mặc định theo thời gian')
args=parser.parse_args()
if args.device=='cuda' and not args.no_launch_blocking: os.environ['CUDA_LAUNCH_BLOCKING']='1'
if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModel, AutoTokenizer
from phase2_core import ROOT, ChunkHead, chunks_from_ids, forward_chunks
from train_baselines import seed_all, restore_rng
from runtime_io import atomic_json

RUN=ROOT/'work_dirs/multichunk_v1'
REPORT_DIR=ROOT/'reports/chunk_attention_v1'
DIAG=REPORT_DIR/'cuda_diagnosis.json'

def nvidia_smi():
    """Ảnh chụp GPU; lỗi khi gọi nvidia-smi được ghi lại, không làm dừng chẩn đoán.

    Trên Windows/WDDM, --query-compute-apps liệt kê cả ứng dụng desktop (C+G), nên chỉ
    lấy tiến trình loại "C" (compute thuần) từ bảng của nvidia-smi để xét GPU có bận không.
    """
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout.strip()
        table=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20).stdout
        compute=[]; graphics=0
        for line in table.splitlines():
            m=re.match(r'\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s+(.+?)\s+\S+\s*\|$',line)
            if not m: continue
            if m.group(2)=='C': compute.append(f'{m.group(1)}, {m.group(3)}')
            else: graphics+=1
        return {'gpu':gpu,'compute_apps':compute,'graphics_apps_count':graphics}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'error':repr(exc)}

def update_diagnosis(replay):
    diag=json.loads(DIAG.read_text(encoding='utf-8')) if DIAG.exists() else {}
    diag.setdefault('replays',[]).append(replay); diag['replay_status']=replay['outcome']
    atomic_json(DIAG,diag)

def main():
    tag=args.tag or time.strftime('%Y%m%d_%H%M%S')+('_cpu' if args.device=='cpu' else '')
    out=REPORT_DIR/'replay'/tag; out.mkdir(parents=True,exist_ok=False)
    before=nvidia_smi() if args.device=='cuda' else None
    others=[a for a in (before or {}).get('compute_apps',[]) if not a.startswith(str(os.getpid()))]
    if args.device=='cuda' and others and not args.allow_shared_gpu:
        result={'tag':tag,'outcome':'deferred_gpu_busy','gpu_before':before,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'note':'GPU có tiến trình compute khác; không chạy song song theo kế hoạch. Dùng --allow-shared-gpu nếu chủ động chấp nhận.'}
        atomic_json(out/'result.json',result); update_diagnosis(result); print(json.dumps(result,ensure_ascii=True)); return 2

    cfg=json.loads((ROOT/'configs/multichunk_pilot.json').read_text())
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['encoder'],local_files_only=True)
    encoder=AutoModel.from_pretrained(ROOT/cfg['encoder'],local_files_only=True,attn_implementation='sdpa').to(device).eval()
    encoder.requires_grad_(False)
    rows=[json.loads(l) for l in (ROOT/cfg['dataset']/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=tok([r['text'] for r in rows],truncation=False,verbose=False)['input_ids']
    train=[{'chunks':chunks_from_ids(s,tok.cls_token_id,tok.sep_token_id),'label':r['label'],'review_id':r['review_id']} for r,s in zip(rows,ids)]

    # Thứ tự khôi phục giống runner: seed_all -> tạo head/optimizer -> nạp state -> restore_rng -> shuffle.
    seed_all(42)
    head=ChunkHead(cfg).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    state=torch.load(RUN/'seed42/latest.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==5
    head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
    if args.device=='cuda': restore_rng(state['rng'])
    else:
        import numpy as np
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy']); torch.set_rng_state(state['rng']['torch'])
    head.train(); order=list(range(len(train))); random.shuffle(order)
    expected=json.loads((REPORT_DIR/'epoch6_order.json').read_text(encoding='utf-8'))
    assert [train[i]['review_id'] for i in order[:2]]==expected[0]['review_ids'], 'Thứ tự epoch 6 không khớp kiểm tra tĩnh'

    window=cfg['batch_size']*cfg['gradient_accumulation']; n=len(order); bs=cfg['batch_size']
    log=(out/'batches.jsonl').open('w',encoding='utf-8')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    started=time.time(); batch_index=0; last=None; outcome='completed'; error=None; loss_sum=0.0; reviews_done=0
    try:
        for ws in range(0,n,window):
            optimizer.zero_grad(set_to_none=True); end=min(ws+window,n)
            for s in range(ws,end,bs):
                if args.max_batches is not None and batch_index>=args.max_batches: raise StopIteration
                batch=[train[i] for i in order[s:min(s+bs,end)]]
                last={'batch':batch_index,'review_ids':[x['review_id'] for x in batch],'chunk_counts':[len(x['chunks']) for x in batch],
                      'chunk_lengths':[[len(c) for c in x['chunks']] for x in batch]}
                logits=forward_chunks(encoder,head,[x['chunks'] for x in batch],device,cfg['chunk_microbatch'],True)
                loss=torch.nn.functional.cross_entropy(logits,torch.tensor([x['label'] for x in batch],device=device),reduction='sum')
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss_sum+=loss.item(); reviews_done+=len(batch); (loss/(end-ws)).backward()
                row={**last,'loss':loss.item(),'t':round(time.time()-started,2)}
                if device.type=='cuda':
                    row['alloc_mib']=round(torch.cuda.memory_allocated()/2**20,1); row['reserved_mib']=round(torch.cuda.memory_reserved()/2**20,1)
                log.write(json.dumps(row)+'\n'); batch_index+=1
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
            if not torch.isfinite(norm): raise RuntimeError(f'Nonfinite grad norm at window {ws}')
            optimizer.step(); log.flush()
    except StopIteration:
        outcome='completed_max_batches'
    except Exception:
        outcome='failed'; error=traceback.format_exc()
    finally:
        log.close()
    peak=None
    if device.type=='cuda':
        try: peak=torch.cuda.max_memory_allocated()/2**20
        except RuntimeError as exc: peak=f'unavailable after CUDA error: {exc.__class__.__name__}'
    result={'tag':tag,'outcome':outcome,'device':args.device,'cuda_launch_blocking':os.environ.get('CUDA_LAUNCH_BLOCKING'),
            'allow_shared_gpu':args.allow_shared_gpu,'batches_completed':batch_index,'epoch6_total_batches':len(expected),
            'mean_train_loss':loss_sum/max(1,reviews_done),'reviews_completed':reviews_done,
            'seconds':round(time.time()-started,1),'last_batch':last if outcome=='failed' else None,'error':error,
            'gpu_before':before,'gpu_after':nvidia_smi() if args.device=='cuda' else None,
            'peak_allocated_mib':peak,
            'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z',time.localtime(started)),
            'note':'Không lưu checkpoint; work_dirs/multichunk_v1 chỉ được đọc.'}
    atomic_json(out/'result.json',result); update_diagnosis(result)
    print(json.dumps({k:result[k] for k in ['tag','outcome','batches_completed','seconds','peak_allocated_mib']},ensure_ascii=True))
    return 1 if outcome=='failed' else 0

if __name__=='__main__': sys.exit(main())
