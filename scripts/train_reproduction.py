"""Faithful PyTorch reproduction of the paper's BERT+BiLSTM (author TF code) on the project's clean IMDb split.

  python scripts/train_reproduction.py --smoke
  python scripts/train_reproduction.py                      (all variants x seeds in run_order; resumable)
  python scripts/train_reproduction.py --variant code --seed 42

Frozen BERT (dropout active while training, as Keras fit does), Dropout(0.5) on token states, BiLSTM(768)
over the full padded sequence, Dense(2)+softmax, Keras binary cross-entropy, Adam with Keras time decay.
No checkpoint selection: the final model after the fixed epoch budget is evaluated on test once.
"""
import argparse
import hashlib
import json
import msvcrt
import os
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from train_baselines import ROOT, encode, evaluate, load_data, save_checkpoint, seed_all
from transformers import AutoModel, AutoTokenizer
from runtime_io import atomic_json, publish_status

CONFIG=ROOT/'configs/reproduction_v1.json'

class ReproHead(nn.Module):
    """Keras graph: Dropout(0.5) -> Bidirectional(LSTM(768)) -> Dense(2, softmax). No mask reaches the LSTM."""
    def __init__(self,units=768,dropout=0.5):
        super().__init__(); self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM(768,units,batch_first=True,bidirectional=True); self.classifier=nn.Linear(2*units,2)
        self.keras_init(units)
    def keras_init(self,units):
        for suffix in ['','_reverse']:
            w_ih=getattr(self.lstm,f'weight_ih_l0{suffix}'); w_hh=getattr(self.lstm,f'weight_hh_l0{suffix}')
            b_ih=getattr(self.lstm,f'bias_ih_l0{suffix}'); b_hh=getattr(self.lstm,f'bias_hh_l0{suffix}')
            with torch.no_grad():
                # Keras initializes the concatenated (input, 4*units) kernel and (units, 4*units) recurrent kernel.
                w_ih.copy_(nn.init.xavier_uniform_(torch.empty(768,4*units)).t())
                w_hh.copy_(nn.init.orthogonal_(torch.empty(units,4*units)).t())
                b_ih.zero_(); b_hh.zero_(); b_ih[units:2*units]=1.0  # unit_forget_bias
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)
    def forward(self,x,mask=None):
        _,(h,_)=self.lstm(self.dropout(x))
        return self.classifier(torch.cat((h[-2],h[-1]),dim=1))

def keras_bce(logits,labels,eps=1e-7):
    p=logits.softmax(-1).clamp(eps,1-eps); y=nn.functional.one_hot(labels,2).to(p.dtype)
    return -(y*p.log()+(1-y)*(1-p).log()).mean()

def scores(y,p): return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'examples':len(y)}

def run_variant(cfg,name,seed,encoder,tokenizer,cache,out,smoke=False):
    v=cfg['variants'][name]; run=out/('smoke' if smoke else '')/f'{name}_seed{seed}'; run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists() and not smoke: return json.loads((run/'result.json').read_text(encoding='utf-8'))
    def status(**kw): publish_status(run/'status.json',{'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'variant':name,'seed':seed,**kw})
    if v['max_length'] not in cache:
        cache[v['max_length']]=load_data({'dataset':cfg['dataset'],'max_length':v['max_length']},tokenizer,smoke,'head')
    datasets,rows,stats=cache[v['max_length']]
    sig=hashlib.sha256((json.dumps(cfg,sort_keys=True)+name+(ROOT/'scripts/train_reproduction.py').read_text(encoding='utf-8')
                        +(ROOT/cfg['dataset']/'manifest.json').read_text(encoding='utf-8')).encode()).hexdigest()
    seed_all(seed); head=ReproHead(cfg['architecture']['bilstm_units'],cfg['architecture']['dropout_on_token_states']).cuda()
    optimizer=torch.optim.Adam(head.parameters(),lr=v['learning_rate'],eps=v['adam_epsilon'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0/(1.0+v['keras_decay']*step))  # Keras legacy `decay`
    order=torch.Generator().manual_seed(seed); history=[]; first=0
    if (run/'latest.pt').exists():
        st=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        if st['signature']!=sig: raise RuntimeError(f'Signature mismatch for {run.name}; start a new version instead')
        head.load_state_dict(st['head']); optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])
        order.set_state(st['order']); torch.set_rng_state(st['torch']); torch.cuda.set_rng_state(st['cuda']); history=st['history']; first=st['epoch']
    epochs=1 if smoke else v['keras_epochs']
    for epoch in range(first,epochs):
        if (out/'PAUSE').exists(): status(stage='paused',completed_epoch=epoch); return None
        started=time.time(); head.train(); encoder.train(); total=0.; seen=0
        for p in range(v['passes_per_epoch']):
            loader=DataLoader(datasets['train'],batch_size=v['batch_size'],shuffle=True,generator=order)
            for step,(ids,mask,labels) in enumerate(loader):
                ids=ids.cuda(); mask=mask.cuda(); labels=labels.cuda()
                loss=keras_bce(head(encode(encoder,ids,mask)),labels)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at epoch {epoch+1}')
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
                total+=loss.item()*len(labels); seen+=len(labels)
                if step%50==0: status(stage='training',epoch=epoch+1,epochs=epochs,pass_index=p+1,passes=v['passes_per_epoch'],
                                      step=step,steps=len(loader),train_loss=total/seen,lr=scheduler.get_last_lr()[0])
        encoder.eval(); val,_=evaluate(encoder,head,datasets['validation'],64)
        history.append({'epoch':epoch+1,'train_loss':total/seen,'validation_accuracy':val['accuracy'],'validation_macro_f1':val['macro_f1'],
                        'validation_loss':val['loss'],'seconds':time.time()-started,'lr':scheduler.get_last_lr()[0]})
        save_checkpoint(run/'latest.pt',{'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'order':order.get_state(),
                                         'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'history':history,'epoch':epoch+1,'signature':sig})
        atomic_json(run/'history.json',history)
        print(json.dumps({'variant':name,'seed':seed,**history[-1]}),flush=True)
    encoder.eval()
    if smoke:
        report={'passed':True,'variant':name,'history':history,'train_examples':len(datasets['train']),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        atomic_json(run/'smoke_results.json',report); return report
    status(stage='final_test',epochs=epochs)
    metrics,pred=evaluate(encoder,head,datasets['test'],64)
    labels=[r['label'] for r in rows['test']]
    groups={g:scores([labels[i] for i,r in enumerate(rows['test']) if r['exceeds_max_length']==flag],
                     [pred[i] for i,r in enumerate(rows['test']) if r['exceeds_max_length']==flag]) for g,flag in [('short',False),('long',True)]}
    with (run/'test_predictions.jsonl').open('w',encoding='utf-8') as f:
        for r,y in zip(rows['test'],pred): f.write(json.dumps({'review_id':r['review_id'],'movie_id':r['movie_id'],'label':r['label'],'prediction':y})+'\n')
    result={'variant':name,'seed':seed,'epochs':epochs,'passes':epochs*v['passes_per_epoch'],'max_length':v['max_length'],
            'test':{'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],'confusion_matrix':metrics['confusion_matrix'],'length_groups':groups},
            'final_validation':history[-1],'truncation':stats,'selection':'final model, no checkpoint selection','training_seconds':sum(h['seconds'] for h in history),'signature':sig}
    save_checkpoint(run/'final.pt',{'head':head.state_dict(),'signature':sig,'variant':name,'seed':seed})
    atomic_json(run/'result.json',result); status(stage='complete',test=result['test'])
    return result

def aggregate(cfg):
    """Summarize every finished run on disk, so partial or per-variant invocations never drop earlier results."""
    report=ROOT/cfg['report_dir']; summary={}
    results=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/cfg['output']).glob('*_seed*/result.json'))]
    for name in cfg['variants']:
        rs=[r for r in results if r and r['variant']==name]
        if not rs: continue
        summary[name]={m:{'mean':float(np.mean([r['test'][m] for r in rs])),'sample_std':float(np.std([r['test'][m] for r in rs],ddof=1)) if len(rs)>1 else None}
                       for m in ['accuracy','macro_f1']}
        summary[name]['seeds']=[r['seed'] for r in rs]
    atomic_json(report/'results.json',{'runs':[r for r in results if r],'aggregate':summary,'paper_reported_accuracy':0.9767,
                                        'note':'Paper accuracy uses its own re-split and evaluation; not directly comparable to the clean split.'})
    return summary

def main(args):
    cfg=json.loads(CONFIG.read_text(encoding='utf-8')); out=ROOT/cfg['output']; out.mkdir(parents=True,exist_ok=True)
    lock=(out/('smoke.lock' if args.smoke else 'runner.lock')).open('a+b'); lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('Another reproduction runner holds the lock') from None
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['encoder'],local_files_only=True)
    encoder=AutoModel.from_pretrained(ROOT/cfg['encoder'],local_files_only=True,attn_implementation='sdpa').cuda().eval(); encoder.requires_grad_(False)
    variants=[args.variant] if args.variant else cfg['run_order']; seeds=[args.seed] if args.seed else cfg['seeds']; cache={}
    for name in variants:
        cache.clear()
        for seed in seeds:
            torch.cuda.reset_peak_memory_stats()
            r=run_variant(cfg,name,seed,encoder,tokenizer,cache,out,args.smoke)
            if r is None: return
            if args.smoke: print(json.dumps(r),flush=True); assert all(p.grad is None for p in encoder.parameters()); break
            aggregate(cfg)
    if not args.smoke and not args.variant and not args.seed: publish_status(out/'status.json',{'stage':'complete','updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--variant',choices=['code','paper']); p.add_argument('--seed',type=int)
    args=p.parse_args()
    try: main(args)
    except Exception:
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        publish_status(ROOT/cfg['output']/'status.json',{'stage':'failed','pid':os.getpid(),'error':traceback.format_exc(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}); raise
