"""Stage 4-5 pilot only: 10 items (5A,5B). Writes under data/custom_batch2/pilot_stage4.
Requires GEMINI_API_KEY in env/.env. No full batch. No protected file changes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

from src.batch2_gemini import GeminiClient, GeminiCallSpec, BATCH2_SEED, ANSWER_SCHEMA, UNANSWERABLE_SCHEMA, parse_json_text
from src.batch2_generation import generate_part_a, generate_part_b
from src.batch2_pipeline import load_passages, load_exclusion_inventory, pilot_selection, bm25_search, dense_search, write_part_csvs
from src.batch2_text import E5_MODEL, E5_QUERY_PREFIX, build_rare_terms, sha256_text
from src.batch2_reports import write_qc_artifacts
from src.retrieval import BM25Index

MODEL = 'gemini-3.1-flash-lite'  # verified available + lowest-cost flash-lite in API list/docs context
THINKING_BUDGET = 0
TEMPERATURE = 0.0
SEED = BATCH2_SEED
OUT = PROJECT_ROOT / 'data/custom_batch2/pilot_stage4'
OUT.mkdir(parents=True, exist_ok=True)

class GeminiProvider:
    model_id = MODEL
    def __init__(self, schema):
        self.schema = schema
        self.usages = []
    def generate_json(self, prompt, schema=None, system=None):
        spec = GeminiCallSpec(model=MODEL, prompt=prompt, temperature=TEMPERATURE, seed=SEED,
                              max_output_tokens=512, response_schema=schema or self.schema,
                              thinking_budget=THINKING_BUDGET, system_instruction=system)
        r = client.generate(spec, use_cache=True)
        self.usages.append({'cached': r.cached, 'cache_key': r.cache_key, **r.usage.__dict__})
        try:
            return parse_json_text(r.text)
        except Exception:
            return {}

def sum_usage(usages):
    def add(key):
        vals=[u.get(key) for u in usages if u.get(key) is not None]
        return sum(vals) if vals else None
    return {'calls': len(usages), 'cached_calls': sum(1 for u in usages if u.get('cached')),
            'input_tokens': add('input_tokens'), 'output_tokens': add('output_tokens'),
            'thinking_tokens': add('thinking_tokens'), 'total_tokens': add('total_tokens')}

print('loading inventory/corpus/indexes...', flush=True)
client = GeminiClient()
# 1-token verification (non-cached result recorded separately)
v = GeminiCallSpec(model=MODEL, prompt='Return JSON {"ok":true}.', temperature=TEMPERATURE, seed=SEED,
                   max_output_tokens=64, response_schema={'type':'OBJECT','properties':{'ok':{'type':'BOOLEAN'}},'required':['ok']},
                   thinking_budget=THINKING_BUDGET)
vr = client.generate(v, use_cache=True)
try: parsed_v=parse_json_text(vr.text)
except Exception: parsed_v={'raw_error':True}
verification = {'model': MODEL, 'cached': vr.cached, 'text': vr.text, 'usage': vr.usage.__dict__, 'parsed': parsed_v}
print('verification', verification['usage'], flush=True)

ex = load_exclusion_inventory(PROJECT_ROOT/'data/custom_batch2/source_inventory.json')
passages = load_passages(PROJECT_ROOT/'data/processed/corpus.jsonl')
unit_parents=[]; passage_text={}
for row in (json.loads(l) for l in (PROJECT_ROOT/'data/processed/corpus_chunks.jsonl').open(encoding='utf-8')):
    unit_parents.append(row.get('parent_id', row['passage_id']))
for p in passages: passage_text[p.passage_id]=p.text
corpus_pairs=[(p.passage_id,p.text) for p in passages if p.source=='miracl']
rare_terms=build_rare_terms([p.text for p in passages if p.source=='miracl'][:5000])
cands=[p for p in passages if p.source=='miracl' and not p.is_short and p.passage_id not in ex['passage_ids'] and p.parent_id not in ex['passage_ids'] and 30<=len(p.text.split())<=180 and p.title]
cands.sort(key=lambda r: sha256_text(f"{SEED}:{r.passage_id}"))
sel = pilot_selection(cands,cands,n_answerable=5,n_unanswerable=5,seed=SEED)
print('pilot ids A', [p.passage_id for p in sel['answerable']], flush=True)
print('pilot ids B', [p.passage_id for p in sel['unanswerable']], flush=True)

bm25 = BM25Index.load(PROJECT_ROOT/'data/cache/bm25_index.pkl')
print('loading dense...', flush=True)
from sentence_transformers import SentenceTransformer
emb = np.load(PROJECT_ROOT/'data/embeddings/corpus_units.f16.npy').astype(np.float32)
encoder = SentenceTransformer(E5_MODEL)

def bm25_audit(query, top_k=20): return bm25_search(bm25, unit_parents, query, top_k=top_k)
def dense_audit(query, top_k=20):
    qv=encoder.encode([E5_QUERY_PREFIX+query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    return dense_search(emb,qv,unit_parents,top_k=top_k)

print('generating Part A...', flush=True)
prov_a = GeminiProvider(ANSWER_SCHEMA)
part_a, ex_a = generate_part_a(sel['answerable'], provider=prov_a, corpus=corpus_pairs, rare_terms=rare_terms,
                               batch_id='pilot4', seed=SEED, bm25_audit=bm25_audit, dense_audit=dense_audit, max_items=5)
print('Part A done', len(part_a), len(ex_a.rows), flush=True)
print('generating Part B...', flush=True)
prov_b = GeminiProvider(UNANSWERABLE_SCHEMA)
part_b, ex_b, audits_b = generate_part_b(sel['unanswerable'], provider=prov_b, passage_text=passage_text,
                                         bm25_audit=bm25_audit, dense_audit=dense_audit, batch_id='pilot4', seed=SEED, top_k=20, max_items=5)
print('Part B done', len(part_b), len(ex_b.rows), flush=True)

all_ex = ex_a.rows + ex_b.rows
write_part_csvs(OUT, part_a=part_a, part_b=part_b, part_c=[])
write_qc_artifacts(OUT/'qc', part_a=part_a, part_b=part_b, part_c=[], exclusions=all_ex, audits={'part_b':audits_b})
summary = {
    'model': MODEL, 'thinking_budget': THINKING_BUDGET, 'temperature': TEMPERATURE, 'seed': SEED,
    'pilot_selection': {'part_a_passages':[p.passage_id for p in sel['answerable']], 'part_b_hints':[p.passage_id for p in sel['unanswerable']]},
    'verification': verification,
    'usage': {'part_a': sum_usage(prov_a.usages), 'part_b': sum_usage(prov_b.usages), 'total_calls_excl_verification_cached': len(prov_a.usages)+len(prov_b.usages)},
    'counts': {'part_a_accepted': len(part_a), 'part_b_accepted': len(part_b), 'excluded': len(all_ex), 'part_c': 0},
    'exclusions': all_ex,
    'part_a_examples': [{k: r.get(k,'') for k in ['question','gold_answer','question_type','gold_passage_id','self_retrieval_gold_rank_bm25','self_retrieval_gold_rank_dense','filter_flags','needs_human_check','my_decision']} for r in part_a],
    'part_b_examples': [{k: r.get(k,'') for k in ['question','question_type','plausible_answerable','hidden_evidence_candidate','top_retrieved_passage_ids','filter_flags','needs_human_check','my_decision']} for r in part_b],
    'audits_b': audits_b,
}
(OUT/'pilot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('WROTE', OUT, flush=True)
print(json.dumps({'counts': summary['counts'], 'usage': summary['usage']}, ensure_ascii=False), flush=True)
