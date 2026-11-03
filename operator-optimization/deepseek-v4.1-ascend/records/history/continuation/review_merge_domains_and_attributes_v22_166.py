from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket,math
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');M=P/'native-PMU-complete-v18';OUT=M/'merge-row-domain-attribute-identity-and-payload-review-v22.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
merge=pin(M/'shared-routed-merge-to-HC-boundary-v22.json','67ea56947962067c41cd3942a6cc37d791dea4db9beec5c609d63eff226fd7db')
attrs=pin(M/'saved-COMPUTE-attribute-field-census-v22.json','46e59d2de1773f4e8f9fd0e717b5ae68a99094644986d1613a8cc8b9caf6f147')
gr=pin(P/'native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991');res=pin(M/'resource-functional-attribution-and-uncertainty-v20-v3.json','c8e2c128d208597aeeb07a2b434506c930397d773b0ab5767a0fadba7e4840c4');origin={(a[0],a[1]):a[2]for a in res['execution_assignments']}
ar={(d['database_index'],r['globalTaskId']):r for d in attrs['databases']for r in d['attribute_records']};pairtypes=Counter();reuse=Counter();examples=[];controlkeys=set();control_occurrences=0
fields=['opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text','blockNum','mixBlockNum']
for r in merge['graph_pairs']:
 di=r['database_index'];a=r['merge']['definition'];b=r['FFN_HcPost']['definition'];aa=ar[(di,a['globalTaskId'])];bb=ar[(di,b['globalTaskId'])]
 pairtypes[json.dumps(dict(merge={k:a[k]for k in fields},HC={k:b[k]for k in fields}),sort_keys=True)]+=1
 reuse['same_raw_attrInfo_ID'if aa['raw_attrInfo']==bb['raw_attrInfo']else'different_raw_attrInfo_ID']+=1
 reuse['same_attribute_text'if aa['string_table_value_at_exact_id']==bb['string_table_value_at_exact_id']else'different_attribute_text']+=1
 control_occurrences+=len(r['overlapping_other_TASK_keys']);controlkeys.update((di,i)for i in r['overlapping_other_TASK_keys'])
 if len(examples)<2:examples.append(dict(database_index=di,graph_ordinal=r['graph_ordinal'],block_ordinal=r['block_ordinal'],merge_attribute=aa,HC_attribute=bb))
def dims(s):return [[int(x)for x in t.split(',')]if t else[]for t in s.strip('"').split(';')]
merge_rows=Counter();target_merge_rows=Counter();payload=graph_payload=0;duration=0
for r in merge['all_merge_shape_context_regimes']:
 s=r['signature'];shape=dims(s['outputShapes_text'])[0];assert len(shape)==2 and shape[1]==5120 and s['outputDataTypes_text']=='DT_BF16';n=r['summary']['executions'];merge_rows[shape[0]]+=n;payload+=math.prod(shape)*2*n;duration+=r['summary']['native_task_duration_sum_ns']
 if r['origin']=='GRAPH':graph_payload+=math.prod(shape)*2*n
 if r['context']=='CALLER_TARGET_MODEL_OR_ENGRAM':target_merge_rows[shape[0]]+=n
 if r['origin']=='QUEUE':assert shape[0]==800;target_merge_rows[shape[0]]+=n
mcrows=Counter();target_native_rows=Counter();mc_regimes=[]
for r in gr['MC2_shape_groups']:
 ss=dims(r['signature']['inputShapes_text']);m=ss[0][0];E=ss[1][0];K=ss[3][-1];orig=Counter(origin[tuple(k)]for k in r['execution_keys']);mcrows[m]+=r['executions'];mc_regimes.append(dict(input_M=m,local_experts=E,TopK=K,executions=r['executions'],origin_counts=dict(orig)))
 if E==24 and K==6:target_native_rows[m]+=orig.get('DIRECT',0)+orig.get('QUEUE',0)
target_native_rows+=Counter();missing_M=sorted(set(target_merge_rows)-set(target_native_rows));assert missing_M==[800]and target_merge_rows[800]==320
assert reuse==Counter(same_raw_attrInfo_ID=3200,same_attribute_text=3200)
assert sum(merge_rows.values())==4472 and payload==8135065600 and graph_payload==98304000
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),graph_pairs=3200,normalized_graph_pair_type_groups=[dict(signature=json.loads(k),pairs=v)for k,v in pairtypes.items()],attribute_reuse_counts=dict(reuse),attribute_examples=examples,attribute_semantic_warning='All3200FFN_HcPost rows reuse the exactattrInfo ID/text oftheprecedingForeachAddList,includinginput_2:1. This doesnot makealpha anHCparameter. Field/source-context association mustbe boundbeforetreatingrawtextasown-opactualattributes;noexporterbugisasserted.',overlapping_control_interval_pair_occurrences=control_occurrences,distinct_overlapping_control_TASK_keys=len(controlkeys),control_interval_warning='Amodel-wait/otherwideinterval canintersectmanylayergaps. Pair-occurrencesare notdistincttasksorremovablewaitcost.',all_merge_row_counts=dict(merge_rows),target_native_merge_row_counts=dict(target_merge_rows),raw_MC2_input_row_counts=dict(mcrows),raw_target_native_MC2_row_counts=dict(target_native_rows),raw_MC2_regimes=mc_regimes,target_merge_row_M_absent_from_raw_MC2=missing_M,affected_target_M800_merge_executions=320,row_domain_conclusion='Naivesame-M replacementofeveryrawMC2outputwiththefinalmergetensor isnotvalidforallcapturedcases.320targetM800mergeoutputs havenoM800rawtargetnativeMC2regime. ExactindividualpaddingpairM816or834hasnotbeenassignedbyordinal/nearesttime.',logical_merge_output_payload_bytes=payload,one_write_one_read_logical_boundary_model_bytes=2*payload,graph_only_output_payload_bytes=graph_payload,graph_only_one_write_one_read_model_bytes=2*graph_payload,all_merge_native_task_duration_sum_ns=duration,payload_limit='Interfacepayload arithmetic only,notactualHBM/networktraffic,a universalI/Olowerbound or causal latencygain. Roundingboundarycanbepreservedinregisterswhilematerializationisaseparatequestion.',no_new_source_DB_model_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
