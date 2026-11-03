from pathlib import Path
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
a=json.loads((P/'runtime/client-baseline/responses.json').read_text());b=json.loads((P/'runtime/client-profile/responses.json').read_text());assert len(a)==len(b)==8
rows=[dict(index=i,visible_text_equal=x['text']==y['text'],baseline_text_sha256=hashlib.sha256(x['text'].encode()).hexdigest(),profile_text_sha256=hashlib.sha256(y['text'].encode()).hexdigest(),baseline_usage=x['usage'],profile_usage=y['usage']) for i,(x,y)in enumerate(zip(a,b))]
raw=(P/'analysis/prepare-phase-and-schedule-cohorts-v12.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='55ab5a2ebb4775f0af61039b2c2ba6e82457f97dcec79fc0033b30c733f77563';c=json.loads(raw)
cohorts=[dict(key=x['key'],ep_ranks=x['ep_ranks'],unique_expected_TP8_group=x['ep_ranks']in[list(range(8)),list(range(8,16))],roots=x['roots']) for x in c['cohorts']]
result=dict(rows=rows,all8_visible_text_equal=all(x['visible_text_equal'] for x in rows),cohorts=cohorts,all11_expected_TP8_group_membership=all(x['unique_expected_TP8_group'] for x in cohorts),
 limits=['Visible text equality is one-round output check,not bitwise intermediate or universal correctness proof.','Cohort membership is based on scheduled runtime IDs/CPUpositions;not GPUvalid masks/crossrank clock alignment.'],no_new_model_work=True,overall_complete=False)
out=P/'analysis/output-and-cohort-checks-v12.json';assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
