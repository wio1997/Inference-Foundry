#!/usr/bin/env python3
"""Check source identities, original-body controls and bounded coverage deltas."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import ast,json
from probe_v1_packet_write_extent import body
ROOT=Path('audit/execution-search')
def load(n):return json.loads((ROOT/(n+'.json')).read_text())
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def main():
    allsources={}
    for n,count in [('v1-input-dependencies',16),('v1-packet-dependencies',11)]:
        d=load(n);assert len(d['sources'])==count and d['remote_writes']==[]
        for s in d['sources']:
            assert sha(s['local_path'])==s['runtime_sha256'];allsources[s['runtime_path']]=s
            if s['local_sha256'] is not None:assert s['matches_local'] is True
    r=load('v1-input-source-review');old=load('whole-forward-coverage-delta');assert len(r['bindings'])==21
    assert len(r['branches'])==156 and Counter(b['condition_state'] for b in r['branches'])==r['branch_states']
    assert sum(b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' for b in r['branches'])==86
    assert sum(a['condition_state']!=b['condition_state'] for a,b in zip(old['branches'],r['branches']))==21
    for a,b in zip(old['branches'],r['branches']):
        for k in ['condition','line','sha256','observed_outcomes','fusion_search_complete']:assert a[k]==b[k]
        assert not b['observed_outcomes'] and b['fusion_search_complete'] is False
    assert len(r['engram_handoff_controls'])==16 and all(not x['cleanup_guard_1529'] for x in r['engram_handoff_controls'])
    prior=r['prior_exact_SHA_references'];assert len(prior)==1 and sha(prior[0]['local_path'])==prior[0]['sha256']
    allsources.update({s['runtime_path']:dict(**s,runtime_sha256=s['sha256']) for s in prior})
    for b in r['bindings']:
        for e in b['evidence']:assert e['sha256']==allsources[e['runtime_path']]['runtime_sha256']
    q=load('v1-qsl-owner-controls');assert q['controls']==32 and len(q['negative_controls'])==3
    assert all(x['task_count']==5 and x['canonical_output_alias_preserved'] for x in q['records'])
    assert any(any(v['QSL_roots']!=['canonical_first_qsl'] for v in x['native_QSL']) for x in q['records'] if not x['canonicalize_incoming'])
    assert all(all(v['QSL_roots']==['canonical_first_qsl'] for v in x['native_QSL']) for x in q['records'] if x['canonicalize_incoming'])
    ex=load('v1-input-source-excerpts');assert len(ex['sources'])==28 and len(ex['excerpts'])==56
    for e in ex['excerpts']:
        p=allsources[e['runtime_path']];assert e['sha256']==p['runtime_sha256'];assert e['lines']==Path(p['local_path']).read_text().splitlines()[e['first_line']-1:e['last_line']]
    packet=load('v1-packet-write-extent');assert len(packet['records'])==12
    assert {(x['operator'],x['tail_words']) for x in packet['records']}=={('SMLA',124),('QLI',160)}
    assert all(x['struct_equal'] and x['tail_sentinels_unchanged'] and not x['full_packet_equal'] for x in packet['records'])
    driver=ROOT/'v1-packet-write-extent-cpu/driver.cpp';assert sha(driver)==packet['driver_sha256'];text=driver.read_text()
    for cls,suf in [('SparseFlashMlaMetadataCpuKernel','/sparse_flash_mla_metadata_aicpu.cpp'),('QuantLightningIndexerV2MetadataCpuKernel','/quant_lightning_indexer_v2_metadata_aicpu.cpp')]:
        s=next(v for k,v in allsources.items() if k.endswith(suf));fn=body(Path(s['local_path']).read_text(),'bool '+cls+'::GenMetadata(');assert text.count(fn)==1
    for n,tool in [('v1-input-dependencies','read_v1_input_dependencies.py'),('v1-input-source-review','review_v1_input_owners.py'),('v1-qsl-owner-controls','probe_v1_qsl_owner_controls.py'),('v1-packet-dependencies','read_v1_packet_boundary.py'),('v1-packet-write-extent','probe_v1_packet_write_extent.py'),('v1-input-parent-price','price_v1_input_parent.py')]:assert load(n)['reader_sha256']==sha('tools/'+tool)
    price=load('v1-input-parent-price');assert len(price['ranks'])==16 and abs(price['mean_three_function_self_Free_ms']-311.050015625)<1e-8
    for n in ['v1-input-source-review','v1-input-parent-price']:
        for x in load(n)['inputs']:assert sha(x['path'])==x['sha256']
    for p in Path('tools').glob('*v1*input*.py'):ast.parse(p.read_text())
    result=dict(status='PASS_BOUNDED_V1_SOURCE_OWNER_SERIALIZER_AND_PRICE_INTEGRITY',current_source_pins=27,prior_exact_SHA_reference=1,source_bindings=70,unclosed_original_sites=86,engram_stub_controls=16,owner_symbolic_controls=32,negative_controls=3,serializer_extent_host_controls=12,bounded_excerpts=56,new_native_admissions=0,whole_chain_complete=False,qualification='Integrity and host/symbolic controls only; strict whole native bytes and all-path execution remain unproved.',reader_sha256=sha(__file__))
    (ROOT/'v1-input-integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
