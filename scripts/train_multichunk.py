"""Validation-gated full IMDb experiment. Never reads the test split."""
import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
import msvcrt
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
from phase2_core import ROOT, ChunkHead, chunks_from_ids, forward_chunks, padded, encoder_features
from train_baselines import Head, seed_all, rng_state, restore_rng, save_checkpoint
from runtime_io import atomic_json, publish_status

def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def scores(y,p):
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)), 'count':len(y)}

def evaluate(encoder,head,items,device,cfg):
    head.eval(); truth=[]; pred=[]; probabilities=[]
    with torch.no_grad():
        for i in range(0,len(items),cfg['batch_size']):
            batch=items[i:i+cfg['batch_size']]
            logits=forward_chunks(encoder,head,[x['chunks'] for x in batch],device,cfg['chunk_microbatch'])
            truth.extend(x['label'] for x in batch); pred.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(-1)[:,1].cpu().tolist())
    result=scores(truth,pred)
    for name,flag in [('long',True),('short',False)]:
        indices=[i for i,x in enumerate(items) if x['long']==flag]
        result[name]=scores([truth[i] for i in indices],[pred[i] for i in indices]) if indices else None
    return result,[{'review_id':x['review_id'],'label':x['label'],'prediction':p,'positive_probability':q,'long':x['long']} for x,p,q in zip(items,pred,probabilities)]

def main(smoke=False):
    cfg=read_json(ROOT/'configs/multichunk_pilot.json'); out=ROOT/cfg['output']; out.mkdir(parents=True,exist_ok=True)
    lock=(out/('smoke.lock' if smoke else 'runner.lock')).open('a+b'); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    def status(**kw):
        publish_status(out/('smoke_status.json' if smoke else 'status.json'),{'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw})
    status(stage='loading')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device('cuda')
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['encoder'],local_files_only=True)
    encoder=AutoModel.from_pretrained(ROOT/cfg['encoder'],local_files_only=True,attn_implementation='sdpa').to(device).eval()
    encoder.requires_grad_(False)
    data={}; stats={}; digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
    for filename in ['train_multichunk.py','phase2_core.py']:
        digest.update((ROOT/'scripts'/filename).read_bytes())
    for split in ['train','validation']:
        path=ROOT/cfg['dataset']/f'{split}.jsonl'; digest.update(path.read_bytes())
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        ids=tok([r['text'] for r in rows],truncation=False,verbose=False)['input_ids']
        items=[]
        for row,seq in zip(rows,ids):
            chunks=chunks_from_ids(seq,tok.cls_token_id,tok.sep_token_id)
            assert [t for chunk in chunks for t in chunk[1:-1]]==seq[1:-1]
            items.append({'chunks':chunks,'label':row['label'],'review_id':row['review_id'],'long':len(seq)>256})
        data[split]=items
        stats[split]={'reviews':len(items),'chunks':sum(len(x['chunks']) for x in items),'max_chunks':max(len(x['chunks']) for x in items),'long_reviews':sum(x['long'] for x in items),'discarded_content_tokens':0}
    signature=digest.hexdigest(); atomic_json(out/'tokenization.json',stats)
    if smoke:
        seed_all(42); head=ChunkHead(cfg).to(device); opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'])
        # One-chunk equivalence with the existing BiLSTM, independent review pooling,
        # gradient flow with checkpointing, and checkpoint/RNG/optimizer restoration.
        head.eval(); reference=Head('bilstm',cfg).to(device).eval(); reference.load_state_dict(head.state_dict())
        seq=data['train'][0]['chunks'][0]; ids,mask=padded([seq],device)
        with torch.no_grad():
            features=encoder_features(encoder,ids,mask,device)
            expected=reference(features,mask)
            actual=head.classify(head.represent(features,mask),torch.tensor([0],device=device),1)
            torch.testing.assert_close(actual,expected)
            reps=torch.randn(3,1536,device=device); owners=torch.tensor([0,0,1],device=device)
            torch.testing.assert_close(head.classify(reps,owners,2),head.classifier(torch.stack([(reps[0]+reps[1])/2,reps[2]])))
        del reference
        batch=[max(data['train'],key=lambda x:len(x['chunks'])),data['train'][0]]
        head.train(); old=head.classifier.weight.detach().clone(); torch.cuda.reset_peak_memory_stats(); started=time.time()
        logits=forward_chunks(encoder,head,[x['chunks'] for x in batch],device,cfg['chunk_microbatch'],True)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([x['label'] for x in batch],device=device)); loss.backward(); opt.step()
        assert torch.isfinite(loss) and not torch.equal(old,head.classifier.weight)
        assert all(p.grad is None for p in encoder.parameters())
        save_checkpoint(out/'smoke.pt',{'head':head.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'signature':signature})
        state=torch.load(out/'smoke.pt',weights_only=False,map_location='cpu'); restored=ChunkHead(cfg).to(device); restored.load_state_dict(state['head'])
        opt2=torch.optim.AdamW(restored.parameters(),lr=cfg['learning_rate']); opt2.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        head.eval(); restored.eval()
        with torch.no_grad(): torch.testing.assert_close(head(features,mask),restored(features,mask))
        # Next update must match after restoring optimizer and dropout RNG.
        state_rng=rng_state(); head.train(); restored.train()
        for model,optimizer in [(head,opt),(restored,opt2)]:
            restore_rng(state_rng); optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(features,mask),torch.tensor([0],device=device)).backward(); optimizer.step()
        for a,b in zip(head.parameters(),restored.parameters()): torch.testing.assert_close(a,b)
        report={'passed':True,'checks':['all content tokens retained','single chunk equivalence','review pooling isolation','longest review backward','frozen encoder','checkpoint prediction equality','optimizer and RNG continuation'],'longest_review_chunks':len(batch[0]['chunks']),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'seconds':time.time()-started,'signature':signature}
        atomic_json(out/'smoke_results.json',report); status(stage='smoke_passed',**report); print(json.dumps(report),flush=True); return
    baseline=[json.loads(l) for l in (ROOT/'reports/validation_analysis/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['review_id'] for r in baseline]==[x['review_id'] for x in data['validation']]
    selection=read_json(ROOT/'reports/validation_analysis/checkpoint_selection.json')['bilstm_headtail']
    assert selection['seed']==42, 'This protocol uses matched seed42 head-tail reference'
    base=scores([x['label'] for x in baseline],[x['predictions']['bilstm_headtail']['label'] for x in baseline])
    long=[x for x in baseline if x['long']]
    base_long=scores([x['label'] for x in long],[x['predictions']['bilstm_headtail']['label'] for x in long])
    for seed in cfg['seeds_if_gate_passes']:
        if seed!=42:
            gate=read_json(out/'pilot_gate.json')
            if not gate['passed']: break
        folder=out/f'seed{seed}'; folder.mkdir(exist_ok=True); seed_all(seed)
        head=ChunkHead(cfg).to(device); optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        history=[]; best=-1.; first=1
        if (folder/'latest.pt').exists():
            state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
            if state['signature']!=signature: raise RuntimeError('Resume signature mismatch; keep this run unchanged')
            head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
            history=state['history']; best=state['best']; first=state['epoch']+1
        for epoch in range(first,cfg['max_epochs']+1):
            if (out/'PAUSE').exists(): status(stage='paused',seed=seed,completed_epoch=epoch-1); return
            head.train(); order=list(range(len(data['train']))); random.shuffle(order); started=time.time(); loss_sum=0
            window=cfg['batch_size']*cfg['gradient_accumulation']; n=len(order)
            for window_start in range(0,n,window):
                optimizer.zero_grad(set_to_none=True); end=min(window_start+window,n)
                for start in range(window_start,end,cfg['batch_size']):
                    batch=[data['train'][i] for i in order[start:min(start+cfg['batch_size'],end)]]
                    logits=forward_chunks(encoder,head,[x['chunks'] for x in batch],device,cfg['chunk_microbatch'],True)
                    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([x['label'] for x in batch],device=device),reduction='sum')
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    loss_sum+=loss.item(); (loss/(end-window_start)).backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); optimizer.step()
                if window_start%(window*10)==0:
                    status(stage='training',seed=seed,epoch=epoch,max_epochs=cfg['max_epochs'],completed_reviews=end,total_reviews=n,train_loss=loss_sum/end,epoch_elapsed_seconds=time.time()-started,best_validation_f1=best)
            status(stage='validation',seed=seed,epoch=epoch,max_epochs=cfg['max_epochs'])
            metrics,predictions=evaluate(encoder,head,data['validation'],device,cfg)
            row={'epoch':epoch,'train_loss':loss_sum/n,'validation':metrics,'seconds':time.time()-started}; history.append(row)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1']; save_checkpoint(folder/'best.pt',{'head':head.state_dict(),'epoch':epoch,'validation':metrics,'signature':signature,'seed':seed})
                atomic_json(folder/'best_validation_predictions.json',predictions)
            save_checkpoint(folder/'latest.pt',{'head':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'history':history,'rng':rng_state(),'signature':signature})
            atomic_json(folder/'history.json',history); print(json.dumps({'seed':seed,**row}),flush=True)
        chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        result={'seed':seed,'epochs_completed':len(history),'best_epoch':chosen['epoch'],'validation':chosen['validation'],'test_evaluated':False,'signature':signature}
        atomic_json(folder/'result.json',result)
        if seed==42:
            metrics=chosen['validation']; delta=metrics['macro_f1']-base['macro_f1']
            passed=delta>=cfg['gate_min_delta_f1'] and metrics['long']['macro_f1']>=base_long['macro_f1']
            atomic_json(out/'pilot_gate.json',{'passed':passed,'baseline_seed':42,'baseline':base,'baseline_long':base_long,'pilot':metrics,'delta_f1':delta,'minimum_delta_f1':cfg['gate_min_delta_f1'],'rule':'overall >= +0.5 percentage points AND long-review F1 does not decrease; practical gate, not significance test'})
            if not passed: status(stage='completed_pilot_gate_not_passed',seed=42,result=result); return
        del head,optimizer; torch.cuda.empty_cache()
    results=[read_json(out/f'seed{s}/result.json') for s in cfg['seeds_if_gate_passes']]
    selected=max(results,key=lambda r:r['validation']['macro_f1'])
    atomic_json(ROOT/'configs/multichunk_deploy.json',{'model':'multichunk','seed':selected['seed'],'epoch':selected['best_epoch'],'validation_macro_f1':selected['validation']['macro_f1'],'checkpoint':str((out/f"seed{selected['seed']}"/'best.pt').relative_to(ROOT))})
    atomic_json(out/'results.json',{'runs':results,'test_evaluated':False}); status(stage='completed',test_evaluated=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    try: main(args.smoke)
    except Exception:
        cfg=read_json(ROOT/'configs/multichunk_pilot.json'); publish_status(ROOT/cfg['output']/('smoke_status.json' if args.smoke else 'status.json'),{'stage':'failed','pid':os.getpid(),'error':traceback.format_exc(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}); raise
