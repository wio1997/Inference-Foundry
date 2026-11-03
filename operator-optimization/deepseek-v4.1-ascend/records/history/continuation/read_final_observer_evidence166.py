import pathlib,json,gzip,collections,statistics,subprocess,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
r=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');run=json.loads((r/'moe-observer-latest.json').read_text());root=pathlib.Path(run['run_root']);summary=json.loads((root/'value-review/summary.json').read_text());finished=json.loads((root/'controller-finished.json').read_text());restore=json.loads((root/'restore-result.json').read_text());assert finished['state']=='CAPTURED_AND_RESTORED' and restore['status']=='PASS_ORIGINAL_BASELINE_RESTORED'
M=collections.Counter();role=collections.defaultdict(list);caps=set();above=0;totals=[];expertmax=0;ordinals=[]
with gzip.open(root/'value-review/decoded-records.jsonl.gz','rt') as f:
 for line in f:
  d=json.loads(line);M[(d['role'],d['M'])]+=1;role[d['role']].append(d['receiver_count_sum']);totals.append(d['receiver_count_sum']);expertmax=max(expertmax,max(d['receiver_expert_counts'],default=0));caps.add(d['capacity_max_output_size']);above+=int(d['receiver_count_sum_above_capacity_argument']);ordinals.append(d['ordinal'])
cohorts=json.loads((root/'value-review/cohorts.json').read_text());examples=[]
for target in ['language_model.model.layers.0.mlp.experts','language_model.model.layers.39.mlp.experts','mtp.0.mlp.experts']:
 for Mvalue in sorted({k[1] for k in M}):
  found=[x for x in cohorts if x['layer']==target and Mvalue in [int(k) for k in x.get('M_histogram',{})]]
  if found:examples.append(found[0])
examples=examples[:12]
shapes=collections.Counter()
for p in (root/'runtime/observer/shape-facts').glob('*.json'):
 d=json.loads(p.read_text());shapes[(d['role'],d['M'],d['E_local'],d['in_profile_run_at_python_entry'],d['capturing'])]+=1
result={'status':summary['status'],'run_root':str(root),'capture':json.loads((root/'capture-result.json').read_text()),'summary':{k:v for k,v in summary.items() if k not in ['matched_cohort_examples','mismatch_examples']},'M_histogram':[{'role':k[0],'M':k[1],'snapshots':v} for k,v in sorted(M.items())],'receiver_count_sum_range':[min(totals),max(totals)],'receiver_count_sum_by_role':{k:{'min':min(v),'median':statistics.median(v),'max':max(v)} for k,v in role.items()},'single_expert_count_max':expertmax,'capacity_arguments':sorted(caps),'snapshots_above_capacity_argument':above,'observed_ordinals_range':[min(ordinals),max(ordinals)],'selected_complete_cohorts':examples,'startup_shape_facts':[{'role':k[0],'M':k[1],'E':k[2],'profile_flag_at_python_entry':k[3],'capturing':k[4],'instances':v} for k,v in sorted(shapes.items())],'original_service_restored':True,'restore_health':restore['health'],'frozen_serve_sha256':json.loads((root/'controller-state.json').read_text())['serve_sha'],'overall_complete':False,'whole_modules_closed':0};(r/'observer-final-evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
