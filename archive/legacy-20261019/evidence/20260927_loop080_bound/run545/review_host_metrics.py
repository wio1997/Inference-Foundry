import pathlib,json,hashlib,collections,statistics,subprocess
root=pathlib.Path('/data/wio/Inference_Foundry');r=root/'evidence/20260927_loop079_identity/run542';out=root/'evidence/20260927_loop080_bound/run545'
rows=[]
for p in sorted((r/'ledger').glob('pid*.jsonl')):
 if '.flush.' not in p.name:rows += [json.loads(l) for l in p.read_text().splitlines()]
orig=[x for x in rows if x['event']=='clock_origin'];assert len(orig)==10
ref=orig[0]
for x in orig:
 assert (x['boot_id'],x['time_namespace'],x['time_namespace_offsets'].split(),x['clock_implementation'],x['clock_resolution_s'])==(ref['boot_id'],ref['time_namespace'],ref['time_namespace_offsets'].split(),ref['clock_implementation'],ref['clock_resolution_s'])
checks={'same_Linux_Host_monotonic_clock_declarations':True,'scope':'Host timestamps only; no controller identity, NPU clock, device-ready or passive timing certificate','phases':{}}
metricpath=root/'evidence/20260927_loop079_identity/run546/host_ledger_metrics.json';metric=json.loads(metricpath.read_text())
for phase in ['warmup','measured']:
 client=json.loads((r/(phase+'_client')/'summary.json').read_text());clock=client['summary']['clock'];assert clock['kernel_boot_id']==ref['boot_id'] and clock['time_namespace']==ref['time_namespace'] and clock['timens_offsets'].split()==ref['time_namespace_offsets'].split() and 'CLOCK_MONOTONIC' in clock['monotonic_info']
 cr={x['response_id']:x for x in client['requests']};rr=[x for x in rows if x.get('phase')==phase];adds={x['request_id']:x for x in rr if x['event']=='output_add_request'};hands={rid:x for x in rr if x['event']=='runtime_handoff' and x['rank']==0 for rid in x['req_ids']};done={rid:x for x in rr if x['event']=='runner_done' and x['rank']==0 for rid in x['req_ids']}
 values=collections.defaultdict(list)
 for rid,a in adds.items():
  c=cr[a['external_req_id']];h=hands[rid];d=done[rid]
  values['client_start_to_output_add_ms'].append((a['monotonic_ns']-c['start_monotonic_ns'])/1e6)
  values['output_add_to_rank0_handoff_ms'].append((h['monotonic_ns']-a['monotonic_ns'])/1e6)
  values['rank0_handoff_to_rank0_done_ms'].append((d['monotonic_ns']-h['monotonic_ns'])/1e6)
  values['rank0_done_to_client_end_ms'].append((c['end_monotonic_ns']-d['monotonic_ns'])/1e6)
 for key,v in values.items():
  expected={'n':len(v),'min':min(v),'median':statistics.median(v),'max':max(v),'sum':sum(v)}
  assert all(abs(metric['phases'][phase][key][k]-val)<1e-6 for k,val in expected.items())
 ordinary=[x for x in rr if x['event']=='scheduler_append' and not x['bulk']];bulk={x['request_id']:x for x in rr if x['event']=='scheduler_append' and x['bulk']}
 g_h={rid:sum(len(x['admitted_raw_ids']) for x in ordinary if x['request_id']==rid and x['after_ns']<=h['monotonic_ns']) for rid,h in hands.items()}
 late=[{'request_id':rid,'G_at_rank0_handoff':g_h[rid],'G_prebulk':bulk[rid]['g_before'],'ordinary_append_after_handoff_ms':[(x['after_ns']-hands[rid]['monotonic_ns'])/1e6 for x in ordinary if x['request_id']==rid and x['admitted_raw_ids'] and x['after_ns']>hands[rid]['monotonic_ns']]} for rid in hands if g_h[rid]!=bulk[rid]['g_before']]
 checks['phases'][phase]={'Run546_timing_summary_independently_matched':True,'G_appended_by_rank0_handoff_sum':sum(g_h.values()),'G_prebulk_sum':sum(x['g_before'] for x in bulk.values()),'late_ordinary_appends':late}
checks['input_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [metricpath,root/'scripts/loop079_host_ledger_metrics_run546.py',root/'evidence/20260927_loop079_identity/run544/astra_posthoc_review.md',root/'evidence/20260927_loop079_identity/run544/audit_results.json']}
(out/'host_metrics_review.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
