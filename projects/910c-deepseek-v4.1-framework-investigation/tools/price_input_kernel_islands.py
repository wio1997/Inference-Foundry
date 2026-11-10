#!/usr/bin/env python3
"""Price exact native islands within the larger input/host parent on old C8 only."""
from pathlib import Path
from hashlib import sha256
from statistics import median
import json,sys,zlib
sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104');ROOT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from price_vocab_selection_fusion import records
from price_attention_send_epilogue import ENDPOINTS
from analyze_final_producer_entry import latest_dp_change
def main():
    receipt=json.loads((ROOT/'input-kernel-windows-receipt.json').read_text());assert not receipt['invalid'] and receipt['active_expansion_count_distribution']=={'1':3624}
    p=Path(receipt['raw_path']);assert sha256(p.read_bytes()).hexdigest()==receipt['raw_sha256'];raw=json.loads(zlib.decompress(p.read_bytes()))
    waves,hist,schedule,_,_=records();offsets=hist['clock_contract']['offset_upper_ns'];by={};active=0
    for r in raw:
        for w in r['windows']:
            if not w['active']:continue
            active+=1;assert all(n==1 for n in w['engram_counts'].values()) and len(w['engram'])==3 and len(w['expansion'])==1
            assert len({t[3] for t in w['engram']})==1
            w['hash_start']=min(t[1] for t in w['engram']);w['hash_end']=max(t[2] for t in w['engram']);by[w['wave'],r['rank']]=w
    costs={};groups=[]
    for wave in waves:
        rows=[]
        for dp in [0,1]:
            ranks=list(range(dp*8,dp*8+8))
            if not all((wave,k) in by for k in ranks):continue
            ws=[by[wave,k] for k in ranks];o=[[offsets[i][j] for j in ranks] for i in ranks]
            hashspan=latest_dp_change([w['hash_end'] for w in ws],[w['hash_start'] for w in ws],o)
            expansion=latest_dp_change([w['expansion'][0][2] for w in ws],[w['expansion'][0][1] for w in ws],o)
            bounds=dict(hash_three_kernel_island=hashspan,expansion_one_kernel=expansion,joint_named_islands=[hashspan[i]+expansion[i] for i in [0,1]])
            rows.append(bounds);groups.append(dict(wave=wave,DP=dp,bounds_ms=bounds))
        if rows:costs[wave]={k:max(row[k][1] for row in rows) for k in rows[0]}
    assert active==3624 and len(groups)==453 and len(costs)==227
    endpoints=json.loads(Path(ENDPOINTS).read_text())['endpoint_ms'];client=schedule['client']
    def price(scope,rho=1.,h=0.):
        out={}
        for metric,base in endpoints.items():
            gains=[]
            for c in client:
                deadline=c['end_ns'] if metric=='makespan' else c['nonempty_chunk_ns'][0]
                selected=[w for w in costs if min(v['entry'] for v in waves[w].values() if 'entry' in v)<=deadline]
                gains.append(sum(costs[w][scope]*rho-h for w in selected))
            saved=max(base)-max(a-b for a,b in zip(base,gains));out[metric]=dict(signed_saved_ms=saved,signed_percent=100*saved/max(base),signed_per_request_ms=gains)
        return out
    gross={k:price(k) for k in next(iter(costs.values()))};scenarios=[dict(scope=k,removable_and_exposed_fraction=rho,new_ms_per_wave=h,result=price(k,rho,h)) for k in gross for rho in [.25,.5,1] for h in [0,.005,.02,.1]]
    caller=ROOT/'common-frame-caller-price.json';old=json.loads(caller.read_text())
    out=dict(status='MATCHED_OLD_C8_NAMED_KERNEL_ISLANDS_AND_SOURCE_POPULATION',active_rank_windows=active,active_TP8_groups=len(groups),active_waves=len(costs),groups=groups,gross=gross,scenarios=scenarios,local_upper_ms={k:dict(n=len(groups),median=median(r['bounds_ms'][k][1] for r in groups),max=max(r['bounds_ms'][k][1] for r in groups)) for k in gross},
        source_population='Source planner13KVgroups/26runnerbuilders/1draftbuilder, followed by original native exact names: each3624 active rank window has exactly1 expansion and1 each Engram request/write/hash kernel. No repeated per-draft-group expansion to delete.',
        scope='Hash island starts first request-index kernel and ends last hash kernel, includes its internal gaps but excludes earlier deadmask/allocation and later gather/lookup/events. Expansion island is one kernel only. These scopes are children of the larger input/Engram/host publication parent, not its whole optimistic upper.',
        decision='Stop repetition-only DSpark common-output deletion under frozen grouping (zero extra calls). Deprioritize independent fixed-dispatch native-only hash/expansion fusion under existing small-scope priority budget after gross/paid review; no new NPU experiment. Keep larger CPU/scalar/coordinate frame, early publisher/ownership and changed cross-consumer layout/megakernel parents pending; no transfer to sixordinary or synthetic group.',
        qualification='Clock-bounded most favorable active DP per wave in same saved fixed-schedule replay only. Gross relaxes all required math/history/cache writes and new work. Signed rho/h are conditional sensitivities, not observed speedup. Not all earlier/later host preparation or table pipeline is included; never stop that parent from this subset.453 active request groups of454physical groups;8synthetic rank windows unpriced.',
        prior_caller_reference=dict(path=str(caller),sha256=sha256(caller.read_bytes()).hexdigest(),qualification='Consulted only after independent source resource plan and current kernel population; not used to choose source group count or source coverage.'),inputs=[dict(path=str(x),sha256=sha256(x.read_bytes()).hexdigest()) for x in [ROOT/'input-resource-source-plan.json',ROOT/'input-kernel-windows-receipt.json',p,Path(ENDPOINTS)]],new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'input-kernel-island-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(gross={k:{m:round(v['signed_percent'],6) for m,v in result.items()} for k,result in gross.items()},local=out['local_upper_ms'],new_NPU_work=0)))
if __name__=='__main__':main()
