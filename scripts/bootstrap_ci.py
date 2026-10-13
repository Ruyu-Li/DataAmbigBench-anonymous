"""95% bootstrap confidence intervals (1,000 resamples of queries) for the main metrics; run after the evaluations."""
import json, random
from collections import defaultdict
from pathlib import Path
import zlib
import numpy as np
B = 1000
def load(paths):
    best = {}
    for p in paths:
        for l in open(p):
            if l.strip():
                x = json.loads(l); k = (x['dataset'], x.get('task','e2e_open'), x['model'], x['id'], x.get('pair_index'))
                if k not in best or x['status'] == 'ok': best[k] = x
    return best
ROOT = Path(__file__).resolve().parents[1] / 'results'
rows = load(list((ROOT / 'hinted').glob('*_results.jsonl')) + list((ROOT / 'open').glob('*_open_results.jsonl')))
metrics = [('detection','dimension_f1','type'),('detection','condition_f1_norm','interp'),('condition_given','answer_relaxed','given'),
           ('e2e','answer_relaxed_norm','e2e'),('e2e_open','multi_answer_rate','surf')]
# per (ds, metric, model): per-query score (CG: mean over the query's pairs, then clusters = queries)
per = defaultdict(lambda: defaultdict(list))
for (ds, task, m, qid, pi), x in rows.items():
    for t, key, name in metrics:
        if task == t:
            per[(ds, name, m)][qid].append(x['scores'][key])
res = {}
for k, d in per.items():
    ids = list(d); vals = [np.array(d[i]) for i in ids]
    sums = np.array([v.sum() for v in vals]); cnts = np.array([len(v) for v in vals])
    point = sums.sum() / cnts.sum()
    rng = np.random.default_rng(zlib.crc32('|'.join(k).encode()))  # order-independent seed per group
    ids = sorted(ids); vals = [np.array(d[i]) for i in ids]
    sums = np.array([v.sum() for v in vals]); cnts = np.array([len(v) for v in vals])
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    res[k] = (100*point, 100*np.percentile(boots, 2.5), 100*np.percentile(boots, 97.5), boots)
# equal-weight average over subsets, bootstrapped jointly
dss = ['sec','nasa','tatqa','retail','ambigdocs','opinionqa','usc','sf311']
avg = {}
for _, _, name in metrics:
    for m in ['gpt-4o-mini','qwen3.6-flash','deepseek-v4-flash']:
        b = np.mean([res[(d, name, m)][3] for d in dss], axis=0); p = np.mean([res[(d, name, m)][0] for d in dss])
        avg[(name, m)] = (p, 100*np.percentile(b, 2.5), 100*np.percentile(b, 97.5))
hw = [ (v[2]-v[1])/2 for v in res.values()]
print('max half-width', round(max(hw),1), 'median', round(float(np.median(hw)),1))
print('max half-width avg', round(max((v[2]-v[1])/2 for v in avg.values()),1))
json.dump({'per': {f'{k[0]}|{k[1]}|{k[2]}': [round(v[0],1), round(v[1],1), round(v[2],1)] for k, v in res.items()},
           'avg': {f'{k[0]}|{k[1]}': [round(v[0],1), round(v[1],1), round(v[2],1)] for k, v in avg.items()}}, open(ROOT / 'bootstrap_ci.json','w'), indent=1)
