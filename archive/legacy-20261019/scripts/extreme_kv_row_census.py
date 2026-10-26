#!/usr/bin/env python3
"""Reuse legal Run84 write-row evidence without guessing KV HBM bytes."""
import argparse,collections,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def stats(a):return dict(median=statistics.median(a),min=min(a),max=max(a),mean=statistics.mean(a))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 rows=[]
 for rank in range(8):
  p=ROOT/f'evidence/20260924_loop035_diagnostic/run84/pages/rank{rank}.json'
  data=json.loads(p.read_text())['rows'];assert len(data)==256
  for r in data:
   checks=r['operator_checks'];by={(c['ratio'],c['layer'].split('.')[-1]):c for c in checks}
   c4=[c for c in checks if c['ratio']==4]
   c128=[c for c in checks if c['ratio']==128]
   assert len(c4)==2 and len(c128)==1
   assert all(c['missing_actual_pages']==0 for c in checks)
   rows.append(dict(rank=rank,cycle=r['cycle'],c4_compressor_valid_rows=c4[0]['valid_rows'],
                    c4_indexer_valid_rows=c4[1]['valid_rows'],
                    c128_compressor_valid_rows=c128[0]['valid_rows'],
                    c4_compressor_layer_rows=c4[0]['valid_rows']*c4[0]['source_layers'],
                    c4_indexer_layer_rows=c4[1]['valid_rows']*c4[1]['source_layers'],
                    c128_compressor_layer_rows=c128[0]['valid_rows']*c128[0]['source_layers'],
                    candidate_page_entries=r['page_count_sum'],metadata_bytes=r.get('metadata_bytes',0)))
 # Restrict steady census to cycles 64:255; startup first-write behavior stays indexed separately.
 steady=[r for r in rows if 64<=r['cycle']<=255]
 keys=['c4_compressor_valid_rows','c4_indexer_valid_rows','c128_compressor_valid_rows',
       'c4_compressor_layer_rows','c4_indexer_layer_rows','c128_compressor_layer_rows',
       'candidate_page_entries','metadata_bytes']
 first_c128={rank:min(r['cycle'] for r in rows if r['rank']==rank and r['c128_compressor_valid_rows']>0) for rank in range(8)}
 out=dict(source='Run84 real-weight legal 12x1024 c12 continuous page audit, eight ranks x256 cycles',
          steady_cycles=[64,255],rank_cycle_count=len(rows),steady_rank_cycle_count=len(steady),
          steady={k:stats([r[k] for r in steady]) for k in keys},first_c128_write_cycle_by_rank=first_c128,
          cache_source_layers=dict(c4_compressor=21,c4_indexer=21,c128_compressor=20),
          source_spec=dict(service_block_size=32,c4_state_dim=2048,c128_state_dim=1024,state_dtype='float32',
                           indexer_k_head_dim=128,indexer_k_dtype='int8 on 910B3',swa_head_dim=512,swa_dtype='bfloat16'),
          interpretation='Valid rows are source-confirmed write cardinalities, not unique bytes read or written; aliasing, padded page layout, cache reuse and overlapping state writes prevent converting them to HBM traffic without a buffer ABI inventory.',
          limitations=['Run84 page audit imposes substantial diagnostic overhead; do not use its TPS/cycle timing as formal capacity.',
                       'Rows cover target and indexer compressed writes but not all SWA or MTP cache access bytes.',
                       'Page entries are candidates for snapshot, not HBM transactions.',
                       'Read traffic of sparse attention and actual cache memory counters remain UNKNOWN.'])
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(steady_rank_cycles=len(steady),c4_rows=out['steady']['c4_compressor_valid_rows'],
                       c128_rows=out['steady']['c128_compressor_valid_rows'],first_c128=first_c128),indent=2))
if __name__=='__main__':main()
