#!/usr/bin/env python3
"""Audit conditional metadata readers and qualify both nonempty/empty old motifs."""
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import zlib

sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records,cached,BODY

def read(p):return json.loads(Path(p).read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    identity=read(OUT/'metadata-readers-execution-identity.json')
    for s in identity['sources']:assert digest(s['local_path'])==s['runtime_sha256']
    receipt=read(OUT/'metadata-producer-tail-receipt.json');assert digest(receipt['raw_path'])==receipt['raw_sha256']
    raw=json.loads(zlib.decompress(Path(receipt['raw_path']).read_bytes()))
    waves,_,_,_,_=records();bodies=cached(BODY)['profiles'];rows=[]
    S='SparseFlashMlaMetadata';Q='VllmAscendQuantLightningIndexerV2Metadata'
    for r in raw:
        rank=r['rank']
        for w in r['windows']:
            old=waves[w['wave']][rank];actual=sum(w['scheduled'].values())
            if w['kind']=='E':
                body=bodies[rank]['bodies'][old['ordinal']]
                per_rank=int(body['HcPre_shape'].strip('"').split(',')[0])
                geometry='actual pinned eager first HcPre local row count; frozen SP+CP TP8'
            else:
                assert actual==24 and w['B']==4
                per_rank=3;geometry='selected fixed B4 verification graph24 rows /TP8'
            start=(rank%8)*per_rank
            query=max(0,min(actual,start+per_rank)-min(actual,start))
            expected=[Q,Q] if query==0 else [S,S,S,Q,Q]
            assert [m[-1] for m in w['metadata']]==expected,(rank,w['wave'],query,w['metadata'])
            tail=w['tail'];assert len(tail)==3 and tail[0][0]==w['metadata'][-1][0]
            assert tail[1][5]=='MEMCPY_ASYNC' and tail[2][5] in ('EVENT RECORD','MEM_WRITE_VALUE')
            assert tail[0][3]==tail[1][3]==tail[2][3]
            assert tail[0][2]<=tail[1][1]<=tail[1][2]<=tail[2][1]<=tail[2][2]
            rows.append(dict(rank=rank,wave=w['wave'],kind=w['kind'],B=w['B'],
                scheduled_actual_tokens=actual,per_rank_partition=per_rank,source_derived_local_query_count=query,
                partition_geometry=geometry,metadata_calls=len(expected),marker=tail[2][5],
                ready_task=tail[2][0],ready_before_body_start_ms=(w['body_start']-tail[2][2])/1e6,
                ready_before_body_end_ms=(w['body_end']-tail[2][2])/1e6))
    assert len(rows)==72
    nonempty=[r for r in rows if r['source_derived_local_query_count']];empty=[r for r in rows if not r['source_derived_local_query_count']]
    assert len(nonempty)==62 and len(empty)==10
    summary={}
    for name,group in [('nonempty',nonempty),('empty',empty),('all',rows)]:
        summary[name]=dict(windows=len(group),min_ready_before_body_start_ms=min(r['ready_before_body_start_ms'] for r in group),
            min_ready_before_body_end_ms=min(r['ready_before_body_end_ms'] for r in group),
            median_ready_before_body_end_ms=median(r['ready_before_body_end_ms'] for r in group))
    frontier=dict(status='PARTIAL_METADATA_READER_AND_COMPLETION_FENCE_REVIEW',whole_chain_complete=False,
        precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',
        source_identity='metadata-readers-execution-identity.json',source_pins=len(identity['sources']),
        reviewed_ranges=[
          dict(file='attention/dsa_v41.py',ranges=[[497,565],[567,683],[690,717],[1120,1281]]),
          dict(file='attention/context_parallel/dsa_v41_cp.py',ranges=[[112,225],[258,350]]),
          dict(file='models/deepseek_v41/indexer.py',ranges=[[175,298]]),
          dict(file='worker/device_metadata.py',ranges=[[74,155],[158,171]]),
          dict(file='worker/model_runner_v1.py',ranges=[[3252,3258]])],
        consumers=[
          dict(kind='COMPRESSOR',read='C2 source writer waits state c2 group before pool_projected /source cos/sin; global cache domain',
               source='dsa_v41.py:523-550',not_a_full_tail_join='earlier stage than QLI INDEXER'),
          dict(kind='ATTENTION',read='local nonzero-query SMLA waits op metadata then reads QSL/seq/physical tables; inverse RoPE and O projection follow',
               source='dsa_v41.py:633-681,704-716',conditional='num_tokens0 skips _forward_attention'),
          dict(kind='INDEXER',read='actual A3 indexer waits QLI metadata then calls V3 with QSL/cache lengths/residual/physical table',
               source='indexer.py:259-289',conditional='zero query or max_cache_seq_len0 returns before wait; non-index-source shares prior topK'),
          dict(kind='CP output',read='pads local query output, restores TP heads, restores FlashComm suffix then O projection',
               source='dsa_v41_cp.py:209-225',conditional='empty query still participates in output exchange/shape protocol')],
        decisive_source_counterexample='A3 build_qli_metadata has no num_actual_tokens0 early return; empty query model skips select/SMLA, and indexer empty cache can also return before wait. release records _buffer_reusable and clears host state; it does not wait every submitted task. Therefore release/all consumers read does not prove a producer-complete fence for a new shared frame.',
        recorded_tail=dict(initial_qualification='62 nonempty motifs qualified;10 failures retained in original receipt, not silently overwritten',
            source_qualified_extension='10 predicted empty-query windows correctly have2QLI and no native3SMLA calls; SMLA zero-fill branch. Eager inherited24-row bucket yields per_rank3: DP0 actual6 empty TP2..7; DP1 actual12 empty TP4..7.',
            summary=summary,windows=rows,
            qualification='Source+unique kernel/copy/ready motif and pinned local geometry, not tensor values or event-handle identity. Frozen C8 only; body end at/before proposed normal-forward join. No new fence executed or overhead measured.'),
        protocol=dict(status='待确认，worth grouped implementation',
            normal='After all tasks and their result copies are queued, record a generation-owned submission_done on producer stream. Join it on default before normal release/any next-frame device overwrite. Keep output-buffer reuse fence and CPU H2D lease separately; do not append later-phase tasks before recording current done.',
            graph='Maintain original external-event/reset semantics and graph addresses; completion fence belongs to this submission, not a reused ready flag from another phase.',
            errors='Before any dispatch use agreed fallback for ineligible branches. After partial submit failure propagate original error and quarantine the frame until actual completion/recreation; never treat release as permission to share/reuse.',
            cost='Observed tails precede body end in72 windows, so no additional device wait is exposed there. New record/wait host/API costs, contention and other workloads still paid in h; no zero-cost claim.'),
        remaining=['full scalar/tensor/alias observer set beyond these principal consumers','native primitive byte identity and graph/error integration for concrete shared producer','frame/guard/object/publication cost and actual removable/exposed fraction','DP packet empty/dummy/K0/error collective sequence and Gloo net cost'],
        next_action='Keep concrete metadata completion/alias/native/cost gaps in the development queue and continue the remaining large final-layer attention/FFN row-demand producer. Do not repeat the no-snapshot probe or claim old CPU-plan gains again.',
        new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])
    p=OUT/'metadata-reader-frontier.json';p.write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_pins=len(identity['sources']),windows=len(rows),nonempty=len(nonempty),empty=len(empty),summary=summary,new_NPU_work=0)))

if __name__=='__main__':main()
