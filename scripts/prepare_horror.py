"""Build an auditable IMDb Horror subset without modifying the source archive."""
import collections
import csv
import gzip
import hashlib
import html
import json
import re
import shutil
import tarfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/imdb_horror_movies_v1'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Prepared dataset already exists; do not silently replace splits.')
    archive = ROOT/'data/raw/aclImdb_v1.tar.gz'
    if not archive.exists():
        shutil.copy2(Path.home()/'Downloads/aclImdb_v1.tar.gz', archive)
    urls = {}
    counts = collections.Counter()
    with tarfile.open(archive, 'r:gz') as tf:
        for m in tf:
            match = re.fullmatch(r'aclImdb/(train|test)/urls_(pos|neg)\.txt', m.name)
            if match:
                urls[match.groups()] = tf.extractfile(m).read().decode().splitlines()
            parts=m.name.split('/')
            if m.isfile() and len(parts)==4 and parts[2] in ('pos','neg','unsup'):
                counts['/'.join(parts[1:3])]+=1
    wanted = set()
    for lines in urls.values():
        for url in lines:
            wanted.add(re.search(r'tt\d+', url).group())
    metadata = ROOT/'data/raw/title.basics.tsv.gz'
    titles={}
    with gzip.open(metadata, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['tconst'] in wanted:
                titles[row['tconst']]={k:row[k] for k in ['titleType','primaryTitle','genres']}
    rows={'train':[], 'test':[]}
    with tarfile.open(archive,'r:gz') as tf:
        for m in tf:
            match=re.fullmatch(r'aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt',m.name)
            if not match: continue
            split,label,idx,rating=match.groups()
            url=urls[(split,label)][int(idx)]
            mid=re.search(r'tt\d+',url).group()
            meta=titles.get(mid,{})
            if 'Horror' not in meta.get('genres','').split(','): continue
            if meta.get('titleType') != 'movie': continue
            raw=tf.extractfile(m).read().decode('utf-8')
            text=html.unescape(re.sub(r'<[^>]+>',' ',raw))
            text=' '.join(text.split())
            assert text
            label_id=int(label=='pos')
            assert (int(rating)>=7 if label_id else int(rating)<=4)
            rows[split].append({'review_id':f'{split}/{label}/{idx}_{rating}',
                'original_split':split,'movie_id':mid,'label':label_id,'rating':int(rating),
                'title':meta['primaryTitle'],'title_type':meta['titleType'],
                'genres':meta['genres'],'text':text,
                'text_hash':hashlib.sha256(text.casefold().encode()).hexdigest()})
    for r in rows.values(): r.sort(key=lambda x:x['review_id'])
    before={s:len(r) for s,r in rows.items()}
    test_movies={r['movie_id'] for r in rows['test']}
    test_hashes={r['text_hash'] for r in rows['test']}
    label_sets=collections.defaultdict(set)
    for r in rows['train']: label_sets[r['text_hash']].add(r['label'])
    conflicts={h for h,labels in label_sets.items() if len(labels)>1}
    removed=[]; clean=[]; seen=set()
    for r in rows['train']:
        reason = ('movie_in_test' if r['movie_id'] in test_movies else
                  'text_in_test' if r['text_hash'] in test_hashes else
                  'conflicting_training_labels' if r['text_hash'] in conflicts else
                  'duplicate_training_text' if r['text_hash'] in seen else None)
        if reason: removed.append({'review_id':r['review_id'],'reason':reason})
        else: clean.append(r); seen.add(r['text_hash'])
    labels=np.array([r['label'] for r in clean]); groups=np.array([r['movie_id'] for r in clean])
    best=None
    for ti,vi in GroupShuffleSplit(n_splits=200,test_size=.15,random_state=42).split(clean,labels,groups):
        if len(set(labels[ti]))<2 or len(set(labels[vi]))<2: continue
        score=abs(len(vi)/len(clean)-.15)+abs(labels[vi].mean()-labels.mean())
        if best is None or score<best[0]: best=(score,ti,vi)
    assert best is not None
    splits={'train':[clean[i] for i in best[1]],'validation':[clean[i] for i in best[2]],'test':rows['test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not ({r['movie_id'] for r in splits[a]} & {r['movie_id'] for r in splits[b]})
        assert not ({r['text_hash'] for r in splits[a]} & {r['text_hash'] for r in splits[b]})
    files={}; stats={}
    for s,rs in splits.items():
        path=OUT/f'{s}.jsonl'
        with path.open('w',encoding='utf-8') as f:
            for r in rs: f.write(json.dumps(r,ensure_ascii=False)+'\n')
        files[s]=sha(path)
        stats[s]={'reviews':len(rs),'movies':len({r['movie_id'] for r in rs}),
                  'labels':dict(collections.Counter(r['label'] for r in rs)),
                  'title_types':dict(collections.Counter(r['title_type'] for r in rs)),
                  'duplicate_text_rows':len(rs)-len({r['text_hash'] for r in rs})}
    report={'source_archive_sha256':sha(archive),'metadata_sha256':sha(metadata),
            'raw_counts':dict(counts),'genre':'Horror','title_type_filter':'movie',
            'missing_metadata_movies':sorted(wanted-set(titles)),
            'before_cleanup':before,'removed_training_rows':removed,
            'splits':stats,'split_sha256':files,'cross_split_movie_overlap':0,
            'cross_split_normalized_text_overlap':0,'validation_selection_seed':42,
            'validation_selection':'200 grouped candidates; closest to 15% review share and training label balance'}
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/data_preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__': main()
