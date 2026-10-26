import csv, glob, json, statistics
from pathlib import Path
root=Path('/data/wio/Inference_Foundry')
files=glob.glob(str(root/'evidence/20260926_loop060_resource/run246/profile/**/kernel_details.csv'),recursive=True)
rows=[]
for f in files:
    rank=int(Path(f).parts[-3].split('_')[0].removeprefix('rank'))
    with open(f) as h:
        for r in csv.DictReader(h):
            if r['Model ID']=='45' and r['Task ID'] in ('2928','2945') and r['Name']=='aclnnInplaceCopy_TensorMoveAiCore_TensorMove':
                rows.append(dict(rank=rank,task_id=int(r['Task ID']),shape=r['Input Shapes'],dtype=r['Input Data Types'],duration_us=float(r['Duration(us)']),read_bytes=float(r['aiv_read_main_memory_datas(KB)'])*1024,write_bytes=float(r['aiv_write_main_memory_datas(KB)'])*1024))
by={}
for tid in (2928,2945):
    rr=[x for x in rows if x['task_id']==tid]
    by[str(tid)]={'count':len(rr),'ranks':sorted(set(x['rank'] for x in rr)),'shape':rr[0]['shape'] if rr else None,'dtype':rr[0]['dtype'] if rr else None,'duration_us_median':statistics.median(x['duration_us'] for x in rr) if rr else None,'read_bytes_median':statistics.median(x['read_bytes'] for x in rr) if rr else None,'write_bytes_median':statistics.median(x['write_bytes'] for x in rr) if rr else None}
source=(Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py')).read_text()
runner=(Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')).read_text()
anchors={'mtp_stash_copy':source.count('self._mtp_hidden_buffer[:num_tokens].copy_('),'mtp_use_guard':'self.speculative_config.method == "mtp" and mtp_hidden_states is not None' in runner,'mtp_buffer_mutable':'("mtp_hidden_buffer", self.model.model._mtp_hidden_buffer)' in runner}
result={'status':'valid' if all(by[str(t)]['count']>=16 for t in (2928,2945)) and anchors['mtp_stash_copy']==2 else 'invalid','sample':by,'anchors':anchors,'interpretation':'Two large graph copies coincide with pre-hc_head MTP stash path; source+shape strongly implicate MTP gather/copy, but only opt-in intervention and graph counter can causally attribute both. 8192×16384 is graph shape, actual frozen target token count 96. Task durations are not additive E2E savings.'}
print(json.dumps(result,indent=2))
