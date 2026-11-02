import collections
import json
import random
import re
import time
from pathlib import Path
from sklearn.metrics import accuracy_score,f1_score,classification_report
from phase2_core import ROOT,Predictor
from runtime_io import atomic_json,publish_status

def main():
    dest=ROOT/'reports/validation_analysis'; dest.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in (ROOT/'data/processed/imdb_full_v1/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    predictor=Predictor('cuda'); atomic_json(dest/'checkpoint_selection.json',predictor.selection)
    outfile=dest/'predictions.jsonl'; predictions=[]
    if outfile.exists():
        predictions=[json.loads(l) for l in outfile.read_text(encoding='utf-8').splitlines()]
        assert [p['review_id'] for p in predictions]==[r['review_id'] for r in rows[:len(predictions)]]
    with outfile.open('a',encoding='utf-8') as f:
        for start in range(len(predictions),len(rows),8):
            batch=rows[start:start+8]; outputs=predictor.predict_many([r['text'] for r in batch])
            for row,out in zip(batch,outputs):
                record={**row,**out}; predictions.append(record); f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
            publish_status(dest/'status.json',{'stage':'validation_inference','completed':len(predictions),'total':len(rows),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    metrics={}
    for kind in ['bilstm','bilstm_headtail']:
        y=[r['label'] for r in predictions]; p=[r['predictions'][kind]['label'] for r in predictions]
        metrics[kind]={'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro'),
                       'classification_report':classification_report(y,p,output_dict=True,zero_division=0)}
    # Diagnostic sample: balanced by prediction outcome and true class, then length.
    # Sampling is diagnostic, so sample category frequencies are not population estimates.
    pools=collections.defaultdict(list)
    for r in predictions:
        a=r['predictions']['bilstm']['label']==r['label']; b=r['predictions']['bilstm_headtail']['label']==r['label']
        outcome='both_wrong' if not a and not b else 'headtail_fixes' if b and not a else 'headtail_harms' if a and not b else 'both_correct'
        r['outcome']=outcome
        pools[(outcome,r['label'],r['long'])].append(r)
    selected=[]; rng=random.Random(20261016)
    for outcome,quota in [('both_wrong',40),('headtail_fixes',20),('headtail_harms',20),('both_correct',20)]:
        eligible=[]
        for label in [0,1]:
            for long in [False,True]:
                pool=pools[(outcome,label,long)][:]; rng.shuffle(pool)
                # Favor readable diagnostic examples <=650 tokens; retain full text.
                pool.sort(key=lambda r:r['token_count']>650)
                chosen=pool[:quota//4]; eligible+=chosen
        if len(eligible)<quota:
            ids={r['review_id'] for r in eligible}
            rest=[r for key,pool in pools.items() if key[0]==outcome for r in pool if r['review_id'] not in ids]
            rng.shuffle(rest); eligible+=rest[:quota-len(eligible)]
        selected+=eligible
    for i,r in enumerate(selected,1):
        text=r['text'].lower()
        r['case_id']=i
        r['screening_flags']={'long_review':r['long'],'negation_cue':bool(re.search(r"\b(not|never|no|hardly|isn't|wasn't|don't|didn't)\b",text)),
             'contrast_cue':bool(re.search(r'\b(but|however|although|despite|yet)\b',text)),
             'horror_vocabulary':bool(re.search(r'\b(terrifying|disturbing|scary|frightening|creepy|horror)\b',text))}
        r['review_status']='pending_assistant_qualitative_review'
    atomic_json(dest/'summary.json',{'validation_only':True,'count':len(rows),'metrics':metrics,
        'outcome_counts':dict(collections.Counter(r['outcome'] for r in predictions)),
        'diagnostic_sample_count':len(selected),'sample_is_representative':False,
        'screening_flags_are_not_verified_error_causes':True})
    atomic_json(dest/'cases_100.json',selected)
    publish_status(dest/'status.json',{'stage':'inference_complete','completed':len(rows),'total':len(rows)})
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
