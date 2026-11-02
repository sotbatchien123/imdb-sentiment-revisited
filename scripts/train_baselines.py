"""Frozen BERT baselines, grouped IMDb splits, resumable epoch checkpoints."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModel, AutoTokenizer
from runtime_io import atomic_json, publish_status

def write_json(path,obj):
    atomic_json(path,obj)

def save_checkpoint(path,obj):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp)
    for attempt in range(20):
        try:
            os.replace(tmp,path); return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.25)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])

class Head(nn.Module):
    def __init__(self,kind,cfg):
        super().__init__(); self.kind='bilstm' if kind.startswith('bilstm') else kind; self.dropout=nn.Dropout(cfg['dropout'])
        if self.kind=='bilstm':
            self.lstm=nn.LSTM(768,cfg['lstm_hidden_size'],batch_first=True,bidirectional=True)
            self.classifier=nn.Linear(2*cfg['lstm_hidden_size'],2)
        else: self.classifier=nn.Linear(768,2)

    def forward(self,x,mask):
        if self.kind=='bilstm':
            x=self.dropout(x)
            packed=nn.utils.rnn.pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            _,(h,_)=self.lstm(packed)
            x=torch.cat((h[-2],h[-1]),dim=1)
        elif self.kind=='mean': x=(x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        else: x=x[:,0]
        return self.classifier(self.dropout(x))

def encode(encoder,ids,mask):
    # no_grad, not inference_mode: trainable head needs to save these tensors for backward.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        x=encoder(input_ids=ids,attention_mask=mask).last_hidden_state
    return x.float()

def select_ids(ids,max_length,cls_id,sep_id,strategy):
    if len(ids)<=max_length: return ids
    budget=max_length-2; content=ids[1:-1]
    if strategy=='headtail':
        left=(budget+1)//2; content=content[:left]+content[-(budget-left):]
    else: content=content[:budget]
    return [cls_id]+content+[sep_id]

def load_data(cfg,tokenizer,smoke=False,strategy='head'):
    result={}; rows_by_split={}; stats={}
    for split in ['train','validation','test']:
        if smoke and split=='test': continue
        rows=[json.loads(l) for l in (ROOT/cfg['dataset']/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        if smoke: rows=rows[:16]
        all_ids=tokenizer([r['text'] for r in rows],add_special_tokens=True,truncation=False,verbose=False)['input_ids']
        lengths=[len(x) for x in all_ids]
        sequences=[]
        for r,ids in zip(rows,all_ids):
            r['exceeds_max_length']=len(ids)>cfg['max_length']
            ids=select_ids(ids,cfg['max_length'],tokenizer.cls_token_id,tokenizer.sep_token_id,strategy)
            sequences.append({'input_ids':ids,'attention_mask':[1]*len(ids)})
        batch=tokenizer.pad(sequences,padding='max_length',max_length=cfg['max_length'],return_tensors='pt')
        result[split]=TensorDataset(batch['input_ids'],batch['attention_mask'],torch.tensor([r['label'] for r in rows]))
        stats[split]={'reviews':len(rows),'truncated_reviews':sum(n>cfg['max_length'] for n in lengths),
                      'truncation_fraction':sum(n>cfg['max_length'] for n in lengths)/len(rows)}
        rows_by_split[split]=rows
    return result,rows_by_split,stats

def evaluate(encoder,head,dataset,batch_size):
    head.eval(); y=[]; pred=[]; losses=[]; started=time.time()
    with torch.no_grad():
        for ids,mask,labels in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            ids=ids.cuda(); mask=mask.cuda(); target=labels.cuda()
            logits=head(encode(encoder,ids,mask),mask)
            losses.append(nn.functional.cross_entropy(logits,target,reduction='sum').item())
            y.extend(labels.tolist()); pred.extend(logits.argmax(1).cpu().tolist())
    metrics={'loss':sum(losses)/len(y),'accuracy':accuracy_score(y,pred),
             'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
             'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
             'classification_report':classification_report(y,pred,labels=[0,1],target_names=['negative','positive'],output_dict=True,zero_division=0),
             'seconds':time.time()-started,'examples':len(y)}
    return metrics,pred

def run(cfg,smoke=False):
    out=ROOT/cfg['output']; out.mkdir(parents=True,exist_ok=True)
    status_file=out/('smoke_status.json' if smoke else 'status.json')
    def status(**kw):
        publish_status(status_file,{'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw})
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this configured run.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    status(stage='loading_encoder')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['encoder'],local_files_only=True)
    encoder=AutoModel.from_pretrained(ROOT/cfg['encoder'],local_files_only=True,attn_implementation='sdpa').cuda().eval()
    encoder.requires_grad_(False)
    status(stage='tokenizing')
    datasets,rows,stats=load_data(cfg,tokenizer,smoke)
    head_data=datasets
    tail_data=load_data(cfg,tokenizer,smoke,'headtail')[0] if 'bilstm_headtail' in cfg['models'] else None
    write_json(out/('smoke_tokenization.json' if smoke else 'tokenization.json'),stats)
    signature=hashlib.sha256((json.dumps(cfg,sort_keys=True)+(ROOT/cfg['dataset']/'manifest.json').read_text(encoding='utf-8')).encode()).hexdigest()
    if smoke:
        checks=[]
        for kind in cfg['models']:
            datasets=tail_data if kind.endswith('headtail') else head_data
            seed_all(42); head=Head(kind,cfg).cuda()
            optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'])
            ids,mask,labels=next(iter(DataLoader(datasets['train'],batch_size=cfg['batch_size'])))
            ids,mask,labels=ids.cuda(),mask.cuda(),labels.cuda()
            before=head.classifier.weight.detach().clone()
            started=time.time(); logits=head(encode(encoder,ids,mask),mask)
            loss=nn.functional.cross_entropy(logits,labels); loss.backward(); optimizer.step()
            assert torch.isfinite(loss) and all(p.grad is None for p in encoder.parameters())
            assert not torch.equal(before,head.classifier.weight)
            # Verify padding cannot affect packed BiLSTM output.
            head.eval()
            if head.kind in ('bilstm','mean'):
                with torch.no_grad():
                    x=torch.randn(2,8,768,device='cuda'); m=torch.tensor([[1]*5+[0]*3,[1]*8],device='cuda')
                    a=head(x,m); x[0,5:]=10000; b=head(x,m)
                    assert torch.allclose(a,b,atol=1e-5)
            with torch.no_grad():
                features=encode(encoder,ids,mask)
                expected=head(features,mask)
                checkpoint=out/f'smoke_{kind}.pt'
                save_checkpoint(checkpoint,{'head':head.state_dict(),'optimizer':optimizer.state_dict()})
                restored=Head(kind,cfg).cuda().eval()
                restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['head'])
                assert torch.allclose(expected,restored(features,mask),atol=1e-5)
                del restored
            val,_=evaluate(encoder,head,datasets['validation'],cfg['batch_size'])
            checks.append({'model':kind,'loss':loss.item(),'head_updates':True,'encoder_gradients_absent':True,
                           'padding_check':head.kind in ('bilstm','mean'),'checkpoint_reload_check':True,'seconds':time.time()-started,
                           'peak_gpu_mb':torch.cuda.max_memory_allocated()/1024**2,'validation_examples':val['examples']})
            del head,optimizer; torch.cuda.empty_cache()
        write_json(out/'smoke_report.json',checks); status(stage='smoke_passed',checks=checks); print(json.dumps(checks,indent=2),flush=True); return
    write_json(out/'run_config.json',{'config':cfg,'signature':signature,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    all_results=[]
    for seed in cfg['seeds']:
        for kind in cfg['models']:
            datasets=tail_data if kind.endswith('headtail') else head_data
            run_dir=out/f'{kind}_seed{seed}'; run_dir.mkdir(exist_ok=True)
            result_path=run_dir/'test_metrics.json'
            if result_path.exists():
                result=json.loads(result_path.read_text()); assert result['signature']==signature
                all_results.append(result); continue
            seed_all(seed); head=Head(kind,cfg).cuda()
            optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            begin=0; best=-1.; bad=0; history=[]
            latest=run_dir/'latest.pt'
            if latest.exists():
                ck=torch.load(latest,map_location='cpu',weights_only=False)
                assert ck['signature']==signature,'Configuration/data changed; use a new output directory.'
                head.load_state_dict(ck['head']); optimizer.load_state_dict(ck['optimizer'])
                begin=ck['epoch']; best=ck['best']; bad=ck['bad']; history=ck['history']; restore_rng(ck['rng'])
            for epoch in range(begin,cfg['max_epochs']):
                if cfg.get('early_stopping',True) and epoch>=cfg.get('min_epochs',0) and bad>=cfg['patience']: break
                if (out/'PAUSE').exists():
                    status(stage='paused',model=kind,seed=seed,next_epoch=epoch+1); return
                head.train(); encoder.eval(); epoch_start=time.time(); total_loss=0.; seen=0
                order_generator=torch.Generator().manual_seed(seed*1000+epoch)
                loader=DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,generator=order_generator)
                optimizer.zero_grad(set_to_none=True)
                total_steps=len(loader); last_status_time=0.
                for step,(ids,mask,labels) in enumerate(loader):
                    ids,mask,labels=ids.cuda(),mask.cuda(),labels.cuda()
                    group_start=(step//cfg['gradient_accumulation'])*cfg['gradient_accumulation']*cfg['batch_size']
                    group_examples=min(cfg['batch_size']*cfg['gradient_accumulation'],len(datasets['train'])-group_start)
                    logits=head(encode(encoder,ids,mask),mask)
                    loss=nn.functional.cross_entropy(logits,labels,reduction='sum')
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    (loss/group_examples).backward()
                    if (step+1)%cfg['gradient_accumulation']==0 or step+1==total_steps:
                        nn.utils.clip_grad_norm_(head.parameters(),1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    total_loss+=loss.item(); seen+=len(labels)
                    if time.time()-last_status_time>=5 or step+1==total_steps:
                        status(stage='training',model=kind,seed=seed,epoch=epoch+1,max_epochs=cfg['max_epochs'],
                               batch=step+1,total_batches=total_steps,train_loss=total_loss/seen,
                               elapsed_epoch_seconds=time.time()-epoch_start,best_validation_macro_f1=best,
                               gpu_allocated_mb=torch.cuda.memory_allocated()/1024**2)
                        last_status_time=time.time()
                status(stage='validation',model=kind,seed=seed,epoch=epoch+1)
                val,_=evaluate(encoder,head,datasets['validation'],cfg['batch_size'])
                improved=val['macro_f1']>best
                if improved:
                    best=val['macro_f1']; bad=0
                    save_checkpoint(run_dir/'best.pt',{'head':head.state_dict(),'epoch':epoch+1,'signature':signature,'validation':val})
                else: bad+=1
                history.append({'epoch':epoch+1,'train_loss':total_loss/seen,'validation':val,'epoch_seconds':time.time()-epoch_start})
                save_checkpoint(latest,{'head':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                                      'signature':signature,'best':best,'bad':bad,'history':history,'rng':rng_state()})
                write_json(run_dir/'history.json',history)
                print(f'{kind} seed={seed} epoch={epoch+1} train_loss={total_loss/seen:.4f} val_f1={val["macro_f1"]:.4f}',flush=True)
            if (out/'PAUSE').exists():
                status(stage='paused',model=kind,seed=seed); return
            ck=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False); head.load_state_dict(ck['head'])
            status(stage='final_test',model=kind,seed=seed,best_epoch=ck['epoch'])
            metrics,pred=evaluate(encoder,head,datasets['test'],cfg['batch_size'])
            metrics.update(model=kind,seed=seed,best_epoch=ck['epoch'],signature=signature,
                           training_seconds=sum(h['epoch_seconds'] for h in history))
            metrics['length_groups']={}
            for group in [False,True]:
                indices=[i for i,r in enumerate(rows['test']) if r['exceeds_max_length']==group]
                if indices:
                    actual=[rows['test'][i]['label'] for i in indices]; predicted=[pred[i] for i in indices]
                    metrics['length_groups']['long' if group else 'short']={'examples':len(indices),'accuracy':accuracy_score(actual,predicted),'macro_f1':f1_score(actual,predicted,average='macro',zero_division=0)}
            with (run_dir/'test_predictions.jsonl').open('w',encoding='utf-8') as f:
                for r,y in zip(rows['test'],pred):
                    f.write(json.dumps({'review_id':r['review_id'],'movie_id':r['movie_id'],'label':r['label'],'prediction':y})+'\n')
            write_json(result_path,metrics); all_results.append(metrics)
            if cfg.get('generate_full_report'):
                from report_full_imdb import generate_report
                generate_report(cfg)
            del head,optimizer; torch.cuda.empty_cache()
    aggregate={}
    for kind in cfg['models']:
        rs=[r for r in all_results if r['model']==kind]
        aggregate[kind]={m:{'mean':float(np.mean([r[m] for r in rs])),'sample_std':float(np.std([r[m] for r in rs],ddof=1))} for m in ['accuracy','macro_f1','training_seconds']}
    write_json(ROOT/cfg.get('report','reports/baseline_results.json'),{'runs':all_results,'aggregate':aggregate,'signature':signature})
    status(stage='complete',aggregate=aggregate)
    if cfg.get('generate_full_report'):
        from report_full_imdb import generate_report
        generate_report(cfg)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--config',default='configs/baseline.json'); args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    out=ROOT/cfg['output']; out.mkdir(parents=True,exist_ok=True)
    # OS releases the lock after process exit, including a crash.
    import msvcrt
    with (out/'runner.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise RuntimeError('A baseline process is already running.')
        try: run(cfg,args.smoke)
        except Exception:
            publish_status(out/('smoke_status.json' if args.smoke else 'status.json'),{'stage':'failed','pid':os.getpid(),'traceback':traceback.format_exc()})
            raise

if __name__=='__main__': main()
