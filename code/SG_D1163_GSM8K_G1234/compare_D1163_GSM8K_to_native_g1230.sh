python3 - <<'PY'
from collections import Counter
from pathlib import Path
import datetime, hashlib, json, socket

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
dataset=root/'client_ds41_g503/quality_g655'
native=dataset/'vllm_native630_full_test'
client=root/'client_ds41_g1122'
sg=client/'SG_Q_policy_g1176_full_test'
freeze=client/'SG_Q_policy_g1176_full_control/frozen_actual_manifest_g1184.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def index(path):
    records=[json.loads(line) for line in path.read_bytes().splitlines()]
    by_request={row['index']:row for row in records}
    assert len(records)==len(by_request)==1319 and set(by_request)==set(range(1319))
    return by_request

manifest=read(freeze)
assert manifest['complete_1319'] and manifest['all_transport_successful']==(manifest['transport_failures']==0)
dataset_id=sha(dataset/'grade_school_math/data/test.jsonl')
grader=sha(dataset/'grade_school_math/dataset.py')
assert manifest['dataset_sha256']==dataset_id and manifest['grader_sha256']==grader
native_result=read(native/'result.json')
assert native_result['completed']==1319 and native_result['correct']==1270
assert native_result['transport_failures']==native_result['length_limited_responses']==0
assert native_result['dataset_sha256']==dataset_id and native_result['grader_sha256']==grader
native_sha=sha(native/'results.jsonl')
assert native_sha==native_result['results_jsonl_sha256']=='b7465931aa775353606b92ba8bb03b2ff4d0b1a2a4aa458e8c28bc2029d56a76'
sg_sha=sha(sg/'results.jsonl')
assert sg_sha==next(row['sha256'] for row in manifest['files'] if row['relative']=='results.jsonl')
left=index(native/'results.jsonl')
right=index(sg/'results.jsonl')
matrix=Counter()
exact=0
skipped=[]
for case,native_row in left.items():
    sg_row=right[case]
    name=f'{case:04d}_request.json'
    p,q=native/name,sg/name
    assert read(p)==read(q) and native_row['expected']==sg_row['expected'],case
    exact += int(sha(p)==sha(q))
    if not (native_row['transport_ok'] and sg_row['transport_ok']) or any(
            row.get('finish_reason')=='length' for row in (native_row,sg_row)):
        skipped.append({'index':case,'native_transport_ok':native_row['transport_ok'],
                        'sg_transport_ok':sg_row['transport_ok'],'native_finish':native_row.get('finish_reason'),
                        'sg_finish':sg_row.get('finish_reason'),'sg_error':sg_row.get('error')})
        continue
    matrix[native_row['correct'],sg_row['correct']]+=1
assert sum(row['correct'] for row in right.values())==manifest['correct']
assert sum(matrix.values())+len(skipped)==1319
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset_SHA256':dataset_id,
        'grader_SHA256':grader,'native_results_SHA256':native_sha,'sg_results_SHA256':sg_sha,
        'sg_frozen_manifest_SHA256':sha(freeze),'requests_compared':1319,'exact_request_file_bytes_equal':exact,
        'paired_untruncated_successful_responses':sum(matrix.values()),
        'paired_grade_counts':{'both_correct':matrix[True,True],'native_only_correct':matrix[True,False],
          'sg_only_correct':matrix[False,True],'neither_correct':matrix[False,False]},
        'excluded_pairs':skipped,'standalone_SG_correct':manifest['correct'],'native_PD_correct':native_result['correct'],
        'scope':'Matching requests and grades, native630 P/D versus D1163 standalone. Final SGLang G1178 P/D quality separate.'}
print('G1230_STANDALONE_GSM8K_PAIRED_GRADES_JSON '+json.dumps(result))
PY
