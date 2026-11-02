"""Shared local inference and review-level multi-chunk model."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from train_baselines import Head, select_ids

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def load_config():
    return json.loads((ROOT/'configs/full_imdb.json').read_text())

def select_validation_checkpoints():
    cfg=load_config(); result={}
    for kind in ['bilstm','bilstm_headtail']:
        candidates=[]
        for seed in cfg['seeds']:
            folder=ROOT/cfg['output']/f'{kind}_seed{seed}'
            history=json.loads((folder/'history.json').read_text())
            best=max(history,key=lambda x:x['validation']['macro_f1'])
            candidates.append({'model':kind,'seed':seed,'epoch':best['epoch'],
                'validation_macro_f1':best['validation']['macro_f1'],
                'checkpoint':str((folder/'best.pt').relative_to(ROOT))})
        result[kind]=max(candidates,key=lambda x:(x['validation_macro_f1'],-x['seed']))
    return result

def chunks_from_ids(ids,cls,sep,content_size=254):
    content=ids[1:-1]
    return [[cls]+content[i:i+content_size]+[sep] for i in range(0,len(content),content_size)] or [[cls,sep]]

def padded(sequences,device):
    longest=max(map(len,sequences))
    ids=torch.zeros(len(sequences),longest,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,seq in enumerate(sequences):
        ids[i,:len(seq)]=torch.tensor(seq,device=device); mask[i,:len(seq)]=1
    return ids,mask

def encoder_features(encoder,ids,mask,device):
    with torch.no_grad():
        if device.type=='cuda':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return encoder(input_ids=ids,attention_mask=mask).last_hidden_state.float()
        return encoder(input_ids=ids,attention_mask=mask).last_hidden_state

class ChunkHead(Head):
    """Shared BiLSTM representation, then equal mean across chunks per review."""
    def __init__(self,cfg): super().__init__('bilstm',cfg)
    def represent(self,x,mask):
        x=self.dropout(x)
        packed=nn.utils.rnn.pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(packed)
        return torch.cat((h[-2],h[-1]),dim=1)
    def classify(self,representations,owners,nreviews):
        pooled=representations.new_zeros((nreviews,representations.shape[-1]))
        pooled.index_add_(0,owners,representations)
        counts=torch.bincount(owners,minlength=nreviews).to(pooled.dtype).unsqueeze(1)
        return self.classifier(self.dropout(pooled/counts.clamp_min(1)))

def forward_chunks(encoder,head,review_sequences,device,microbatch=4,checkpoint_head=False):
    flat=[]; owners=[]
    for owner,chunks in enumerate(review_sequences):
        flat.extend(chunks); owners.extend([owner]*len(chunks))
    reps=[]
    for start in range(0,len(flat),microbatch):
        ids,mask=padded(flat[start:start+microbatch],device)
        features=encoder_features(encoder,ids,mask,device)
        if checkpoint_head and head.training:
            from torch.utils.checkpoint import checkpoint
            reps.append(checkpoint(head.represent,features,mask,use_reentrant=False))
        else: reps.append(head.represent(features,mask))
    return head.classify(torch.cat(reps),torch.tensor(owners,device=device),len(review_sequences))

class Predictor:
    def __init__(self,device='cpu'):
        torch.set_num_threads(4)
        self.device=torch.device(device); self.cfg=load_config()
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/self.cfg['encoder'],local_files_only=True)
        self.encoder=AutoModel.from_pretrained(ROOT/self.cfg['encoder'],local_files_only=True,attn_implementation='sdpa').to(self.device).eval()
        self.encoder.requires_grad_(False)
        self.selection=select_validation_checkpoints(); self.heads={}
        for kind,entry in self.selection.items():
            head=Head(kind,self.cfg).to(self.device).eval()
            ck=torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=False)
            head.load_state_dict(ck['head']); self.heads[kind]=head
        deploy=ROOT/'configs/multichunk_deploy.json'
        if deploy.exists():
            entry=json.loads(deploy.read_text()); head=ChunkHead(self.cfg).to(self.device).eval()
            head.load_state_dict(torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=False)['head'])
            self.heads['multichunk']=head; self.selection['multichunk']=entry

    def predict_many(self,texts,kinds=None):
        kinds=kinds or ['bilstm','bilstm_headtail']
        tokenizer=self.tokenizer
        original=tokenizer(texts,truncation=False,verbose=False)['input_ids']
        output=[{'token_count':len(ids),'long':len(ids)>256,'predictions':{}} for ids in original]
        with torch.no_grad():
            for kind in kinds:
                if kind=='multichunk':
                    seq=[chunks_from_ids(ids,tokenizer.cls_token_id,tokenizer.sep_token_id) for ids in original]
                    logits=forward_chunks(self.encoder,self.heads[kind],seq,self.device)
                else:
                    seq=[select_ids(ids,256,tokenizer.cls_token_id,tokenizer.sep_token_id,'headtail' if kind.endswith('headtail') else 'head') for ids in original]
                    ids,mask=padded(seq,self.device)
                    logits=self.heads[kind](encoder_features(self.encoder,ids,mask,self.device),mask)
                probabilities=logits.softmax(-1).cpu().tolist()
                for out,p in zip(output,probabilities):
                    out['predictions'][kind]={'label':int(p[1]>=p[0]),'positive_probability':p[1]}
        return output

    def demo(self,text):
        kinds=list(self.heads); result=self.predict_many([text],kinds)[0]
        ids=self.tokenizer(text,truncation=False,verbose=False)['input_ids']
        shown={}
        for kind in kinds:
            if kind=='multichunk':
                chunks=chunks_from_ids(ids,self.tokenizer.cls_token_id,self.tokenizer.sep_token_id)
                shown[kind]={'segments':[self.tokenizer.decode(c,skip_special_tokens=True) for c in chunks],
                             'retained_content_tokens':len(ids)-2,'discarded_content_tokens':0}
            else:
                kept=select_ids(ids,256,self.tokenizer.cls_token_id,self.tokenizer.sep_token_id,'headtail' if kind.endswith('headtail') else 'head')
                if len(ids)>256 and kind.endswith('headtail'):
                    segments=[self.tokenizer.decode(kept[1:128]),self.tokenizer.decode(kept[128:-1])]
                else: segments=[self.tokenizer.decode(kept,skip_special_tokens=True)]
                shown[kind]={'segments':segments,'retained_content_tokens':len(kept)-2,'discarded_content_tokens':len(ids)-len(kept)}
        result.update(visible_input=shown,selection=self.selection,
                      note='Tokenized text is reconstructed for inspection; scores are uncalibrated model probabilities.')
        return result
