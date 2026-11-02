import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runtime_io import atomic_json, publish_status
from train_baselines import select_ids

ROOT=Path(__file__).resolve().parents[1]

class Checks(unittest.TestCase):
    def test_headtail_preserves_both_ends(self):
        ids=[101]+list(range(1000,2000))+[102]
        result=select_ids(ids,256,101,102,'headtail')
        self.assertEqual(len(result),256)
        self.assertEqual(result,[101]+list(range(1000,1127))+list(range(1873,2000))+[102])
        self.assertEqual(select_ids([101,7,102],256,101,102,'headtail'),[101,7,102])

    def test_status_lock_does_not_stop_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
            with patch('runtime_io.os.replace',side_effect=PermissionError('simulated Windows reader lock')):
                with contextlib.redirect_stdout(io.StringIO()):
                    publish_status(Path(td)/'status.json',{'stage':'training'})

    def test_atomic_retry_recovers(self):
        import os
        original=os.replace; calls=[]
        def flaky(a,b):
            calls.append(1)
            if len(calls)==1: raise PermissionError('transient lock')
            return original(a,b)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
            p=Path(td)/'check.json'
            with patch('runtime_io.os.replace',side_effect=flaky): atomic_json(p,{'ok':True})
            self.assertTrue(json.loads(p.read_text())['ok'])

    def test_splits_preserve_official_test_without_leakage(self):
        base=ROOT/'data/processed/imdb_full_v1'
        data={s:[json.loads(l) for l in (base/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ['train','validation','test']}
        self.assertEqual(len(data['test']),25000)
        self.assertTrue(all(r['original_split']=='test' for r in data['test']))
        for s in ['train','validation']:
            self.assertTrue(all(r['original_split']=='train' for r in data[s]))
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            for field in ['movie_id','text_hash']:
                self.assertFalse({r[field] for r in data[a]} & {r[field] for r in data[b]})

if __name__=='__main__': unittest.main(verbosity=2)
