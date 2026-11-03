"""Complete per-role/shape aggregate comparison; per-call pairing not required."""
import collections
import hashlib
import json
from pathlib import Path
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
O=A/'joint170';native=json.loads((O/'F21-SHAPE-AGGREGATES.json').read_text());by_pid={p['pid']:p for p in native['profiles']};source=collections.defaultdict(lambda:{'source_records':0,'receiver_rows':0,'direct_MAC':0});native_group=collections.defaultdict(lambda:{'native_records':0,'native_work_ns':0,'captured_records':0});inputs=[]
for p in sorted((A/'joint168/derived').glob('*-rows.jsonl')):
 b=p.read_bytes();inputs.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 for line in b.splitlines():
  r=json.loads(line);assert r['pid'] in by_pid;role=r['role'];e=24 if role=='target' else 8;assert r['E_local']==e
  k=(r['pid'],role,r['M']);s=source[k];s['source_records']+=1;s['receiver_rows']+=r['receiver_rows'];s['direct_MAC']+=r['receiver_rows']*35389440
for p in native['profiles']:
 for g in p['groups']:
  shapes=g['input_shapes'].strip('"').split(';');x=list(map(int,shapes[0].split(',')));w1=list(map(int,shapes[1].split(',')));w2=list(map(int,shapes[2].split(',')));assert x[1]==5120 and w1[1:]==[5120,4608] and w2[1:]==[2304,5120] and w1[0]==w2[0] and w1[0] in [24,8]
  role='target' if w1[0]==24 else 'draft';k=(p['pid'],role,x[0]);n=native_group[k];n['native_records']+=g['calls'];n['native_work_ns']+=g['native_work_ns'];n['captured_records']+=g['calls'] if g['captured'] else 0
assert set(source)==set(native_group)
details=[];aggregate=collections.defaultdict(lambda:collections.Counter())
for key,s in sorted(source.items()):
 n=native_group[key];assert s['source_records']==n['native_records'];row={'pid':key[0],'role':key[1],'M':key[2],**s,**n};details.append(row);bucket=('target_small' if key[2]<=3 else 'target_prefill') if key[1]=='target' else 'draft'
 for field in ['source_records','receiver_rows','direct_MAC']:aggregate[bucket][field]+=s[field]
 for field in ['native_records','native_work_ns','captured_records']:aggregate[bucket][field]+=n[field]
for bucket,a in aggregate.items():a['direct_MAC_per_cumulative_kernel_second']=a['direct_MAC']/(a['native_work_ns']/1e9)
ratio=aggregate['target_prefill']['direct_MAC_per_cumulative_kernel_second']/aggregate['target_small']['direct_MAC_per_cumulative_kernel_second']
assert aggregate['target_small']['source_records']==145280 and aggregate['target_prefill']['source_records']==2560 and aggregate['draft']['source_records']==11088
assert sum(a['native_records'] for a in aggregate.values())==158928
assert sum(a['native_work_ns'] for a in aggregate.values())==64093352625
r={'status':'REVIEWED_COMPLETE_ROLE_SHAPE_AGGREGATE_WORK_TIME_IN_SOURCE_DOMAIN','source_history_inputs':inputs,'per_pid_role_M_buckets':details,'aggregate':dict(aggregate),'target_prefill_over_small_row_direct_MAC_time_efficiency_ratio':ratio,
 'coverage_reasoning':['All16 existing process histories and all16 runtime nativeF21 inventories cover the same controlledCase boundaries;v168 validated no missing/extra/overflow histories.','Each source target uses24localexperts and each draft uses8. AllnativeW1/W2shapes validate corresponding E,5120,4608 andE,2304,5120. Othercallers/instrumentation do not add anF21 inthiscohort.','Every(pid,role,M) bucket has equal source/native cardinality;there are no extra/missingbuckets. Sumreceiverwork and sumkernel time in eachcompletebucket are valid without identifying individualcalls or matchingcrossrankordinals.','Allobservedcapacity65536hasno countaboveclip;the pinned directdense source algorithm requires receiver_rows*35389440 MAC. Applyingthat source/algorithm domain to observedcounts assumes returnedcounter snapshots obey the current output/stream contract;not a per-native pointer/binary certificate.'],
 'what_is_not_required':['Eager per-call native/history pairing for these aggregate totals.','Cross-rankcall/communicator alignment for a sum over all local work/time within the same cohort.'],
 'normalized_reference_capacity_comparison':{'formula':'eta_ref(bucket)=Q_MAC/(P_ref*sum_native_F21_time),withthe samepositiveper-deviceP_ref inboth targetbuckets.','ratio_cancels_unknown_P_ref':ratio,'meaning':'Targetprefill has thismanytimes the direct-MAC-per-cumulative-full-F21-time efficiency of targetsmallrows inthisdiagnosticcohort.','domain':'Same directINT8denseMAC algorithm and commonconstantper-device referencecapacity. Changing clocks/poolassignment oralgorithm requiresnewcapacitymodel.','not_claimed':'An actual physicalMFU number,wholemodulelowest,cluster/request throughput,achievable speedup or a communication rootcause.'},
 'still_open':['Per-call eager association andwaitingcauses,ifneededforaspecificquestion.','Actualinstructionmode/loadedbinary/physicalcapacityupper andproductiontiming.','Allreader/numerics/newfusionlegality andwholemodules.'], 'new_per_call_eager_associations':0,'new_whole_module_closures':0,'overall_complete':False}
f=O/'MC2-AGGREGATE-WORK-TIME-REVIEW.json';f.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
