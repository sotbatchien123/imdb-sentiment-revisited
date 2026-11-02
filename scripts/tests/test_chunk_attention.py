"""CPU unit tests for plan C pooling heads (no BERT needed).

  .venv-train\\Scripts\\python.exe -m unittest discover -s scripts\\tests -v
"""
import io
import random
import sys
import types
import unittest
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chunk_attention_model import (ChunkCompareHead, accumulation_windows, attention_stats, build_head,
                                   forward_reviews, shared_init_state, state_hash)

CFG={'lstm_hidden_size':8,'dropout':0.5,'attention_dim':4,'attention_init_seed_offset':1000}
DIM=2*CFG['lstm_hidden_size']

class FakeEncoder(nn.Module):
    """Deterministic frozen stand-in for BERT: token embedding as last_hidden_state."""
    def __init__(self):
        super().__init__(); torch.manual_seed(0); self.emb=nn.Embedding(64,768); self.requires_grad_(False)
    def forward(self,input_ids,attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))

def heads(seed=42):
    shared=shared_init_state(CFG,seed)
    return build_head(CFG,'mean',seed,shared).eval(),build_head(CFG,'attention',seed,shared).eval()

def reviews():
    return [[[1,5,6,7,2],[1,8,9,2]],[[1,10,11,12,13,2]],[[1,3,4,2],[1,14,2],[1,15,16,17,2]]]

class PoolingTests(unittest.TestCase):
    def test_single_chunk_mean_equals_attention(self):
        mean,att=heads(); reps=torch.randn(1,DIM); owners=torch.tensor([0])
        torch.testing.assert_close(mean(reps,owners,1),att(reps,owners,1))
        _,w=att(reps,owners,1,return_weights=True); torch.testing.assert_close(w,torch.ones(1))

    def test_uniform_scores_equal_mean(self):
        mean,att=heads(); nn.init.zeros_(att.attention_vector.weight)
        reps=torch.randn(5,DIM); owners=torch.tensor([0,0,1,1,1])
        torch.testing.assert_close(att(reps,owners,2),mean(reps,owners,2))

    def test_weights_valid_per_review(self):
        _,att=heads(); reps=torch.randn(7,DIM)*10; owners=torch.tensor([0,0,0,1,2,2,2])
        _,w=att(reps,owners,3,return_weights=True)
        self.assertTrue(torch.isfinite(w).all() and (w>=0).all())
        torch.testing.assert_close(torch.zeros(3).index_add(0,owners,w),torch.ones(3))

    def test_review_isolation(self):
        for head in heads():
            reps=torch.randn(5,DIM); owners=torch.tensor([0,0,1,1,1])
            alone=head(reps[:2],torch.tensor([0,0]),1)
            together=head(reps,owners,2)[:1]
            torch.testing.assert_close(alone,together)

    def test_permutation_within_review(self):
        _,att=heads(); reps=torch.randn(4,DIM); owners=torch.zeros(4,dtype=torch.long); perm=torch.tensor([2,0,3,1])
        pooled,w=att.pool(reps,owners,1); pooled2,w2=att.pool(reps[perm],owners,1)
        torch.testing.assert_close(pooled,pooled2); torch.testing.assert_close(w[perm],w2)

    def test_gradients_reach_head_not_encoder(self):
        encoder=FakeEncoder(); _,att=heads(); att.train()
        logits=forward_reviews(encoder,att,reviews(),torch.device('cpu'),microbatch=2,checkpoint_head=True)
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,0])).backward()
        for name,p in att.named_parameters(): self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_microbatch_invariance(self):
        encoder=FakeEncoder()
        for head in heads():
            a=forward_reviews(encoder,head,reviews(),torch.device('cpu'),microbatch=1)
            b=forward_reviews(encoder,head,reviews(),torch.device('cpu'),microbatch=4)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

class InitAndLoopTests(unittest.TestCase):
    def test_shared_init_identical_and_rng_untouched(self):
        torch.manual_seed(7); before=torch.get_rng_state()
        mean,att=heads(); shared=shared_init_state(CFG,42)
        torch.testing.assert_close(torch.get_rng_state(),before)
        for key in shared:
            torch.testing.assert_close(mean.state_dict()[key],att.state_dict()[key])
        self.assertEqual(state_hash(shared),state_hash(shared_init_state(CFG,42)))
        self.assertNotEqual(state_hash(shared),state_hash(shared_init_state(CFG,43)))

    def test_order_generator_independent_of_init(self):
        order=random.Random(42+2000); first=list(range(20)); order.shuffle(first)
        order=random.Random(42+2000); heads(); second=list(range(20)); order.shuffle(second)
        self.assertEqual(first,second)

    def test_accumulation_windows_cover_all_and_normalize_last(self):
        windows=list(accumulation_windows(101,2,16))
        self.assertEqual([s for s,_ in windows],[32,32,32,5])
        covered=[i for _,b in windows for s,e in b for i in range(s,e)]
        self.assertEqual(covered,list(range(101)))
        self.assertEqual(windows[-1][1],[(96,98),(98,100),(100,101)])

    def test_attention_stats_skip_single_chunk(self):
        w=torch.tensor([1.0,0.5,0.5,0.9,0.1]); owners=torch.tensor([0,1,1,2,2])
        ent,peak=attention_stats(w,owners,3)
        self.assertEqual(len(ent),2); self.assertAlmostEqual(ent[0],1.0,places=6); self.assertAlmostEqual(peak[1],0.9,places=6)

    def test_checkpoint_resume_next_step_matches(self):
        encoder=FakeEncoder(); shared=shared_init_state(CFG,42); device=torch.device('cpu')
        a=build_head(CFG,'attention',42,shared); opt_a=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=0.01)
        def step(model,opt):
            model.train(); opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(forward_reviews(encoder,model,reviews(),device,2,True),torch.tensor([0,1,0])).backward(); opt.step()
        step(a,opt_a)
        # Round-trip through torch.save like a real resume; loading a live state_dict would share optimizer tensors.
        buffer=io.BytesIO(); torch.save({'head':a.state_dict(),'optimizer':opt_a.state_dict()},buffer); buffer.seek(0)
        saved=torch.load(buffer,weights_only=False)
        b=build_head(CFG,'attention',42,shared); b.load_state_dict(saved['head'])
        opt_b=torch.optim.AdamW(b.parameters(),lr=1e-3,weight_decay=0.01); opt_b.load_state_dict(saved['optimizer'])
        rng=torch.get_rng_state()
        for model,opt in [(a,opt_a),(b,opt_b)]: torch.set_rng_state(rng); step(model,opt)
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q)

    def test_unknown_aggregation_rejected(self):
        with self.assertRaises(ValueError): ChunkCompareHead(CFG,'max')

if __name__=='__main__': unittest.main()
