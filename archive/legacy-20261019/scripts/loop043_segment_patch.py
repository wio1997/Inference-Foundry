#!/usr/bin/env python3
"""Temporary actual-DSpark four-segment host/NPU event probe."""
import argparse,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=[Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py'),
       root/'bootstrap/vllm_dspark_handoff.py',root/'runtime/fixed_serving.py']
def sha(b):return hashlib.sha256(b).hexdigest()
def once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new,1)
def patch_proposer(s):
    s=once(s,'import copy\nfrom collections.abc', 'import copy\nimport time\nfrom collections.abc')
    a=s.index('    def _run_merged_draft(')
    b=s.index('    def set_inputs_first_pass(',a)
    body=s[a:b]
    body=once(body,'        model_input_ids = self.input_ids[:num_input_tokens]', '''        phase = None
        if getattr(self, "_loop043_capture_enabled", False):
            index = self._loop043_cycle
            self._loop043_cycle += 1
            if 64 <= index < 128:
                phase = {"cycle": index, "marks": []}
        def segment_mark(label):
            if phase is not None:
                event = torch.npu.Event(enable_timing=True)
                event.record()
                phase["marks"].append((label, time.perf_counter_ns(), time.thread_time_ns(), event))
        segment_mark("begin")
        model_input_ids = self.input_ids[:num_input_tokens]''')
    body=once(body,'            self.build_model_inputs_first_pass(num_input_tokens, self._context_slot_mapping_buffers)',
       '            self.build_model_inputs_first_pass(num_input_tokens, self._context_slot_mapping_buffers)\n            segment_mark("post_context_kv")')
    assert body.count('        ret_hidden_states = self.model(**model_kwargs)') == 2
    body=body.replace('        ret_hidden_states = self.model(**model_kwargs)',
       '        ret_hidden_states = self.model(**model_kwargs)\n        segment_mark("post_model")',1)
    body=once(body,'                    raw_logits = self.model.compute_logits(sample_hidden_states)',
       '                    segment_mark("pre_lmhead")\n                    raw_logits = self.model.compute_logits(sample_hidden_states)\n                    segment_mark("post_lmhead")')
    body=once(body,'                        draft_token_ids[:, idx + 1].copy_(logits[:, idx].argmax(dim=-1))',
       '                        draft_token_ids[:, idx + 1].copy_(logits[:, idx].argmax(dim=-1))\n                    segment_mark("post_markov")')
    body=once(body,'                return draft_token_ids[:, 1:]',
       '                if phase is not None:\n                    self._loop043_rows.append(phase)\n                return draft_token_ids[:, 1:]')
    return s[:a]+body+s[b:]
def patch_handoff(s):
    return once(s,'        self.proposer.runner = _DP1RunnerShim(inputs)', '''        self.proposer.runner = _DP1RunnerShim(inputs)
        if os.getenv("EXTREME_DSPARK_SEGMENTS_DIR"):
            self.proposer._loop043_capture_enabled = True
            self.proposer._loop043_cycle = 0
            self.proposer._loop043_rows = []''')
def patch_serving(s):
    marker='        dag_dir = os.getenv("EXTREME_RUNTIME_DAG_DIR")'
    added='''        segment_dir = os.getenv("EXTREME_DSPARK_SEGMENTS_DIR")
        if segment_dir:
            torch.npu.synchronize()  # after the timed cohort
            rank = torch.distributed.get_rank()
            path = Path(segment_dir)
            path.mkdir(parents=True, exist_ok=True)
            rows = []
            for row in self.runtime.proposer.proposer._loop043_rows:
                marks = row["marks"]
                rows.append({"cycle": row["cycle"],
                    "marks": [{"name": m[0], "wall_ns": m[1], "thread_cpu_ns": m[2]} for m in marks],
                    "device_stage_ms": {b[0]: a[3].elapsed_time(b[3]) for a,b in zip(marks,marks[1:])}})
            (path / f"rank{rank}.json").write_text(json.dumps({
                "rank": rank, "cycles": cycles, "rows": rows,
                "capture_window": [64, 127],
                "clock": "Host perf_counter/thread_time and NPU events; synchronize after cohort only",
            }))
'''
    return once(s,marker,added+marker)
parser=argparse.ArgumentParser();parser.add_argument('action',choices=('install','restore'));parser.add_argument('--record',required=True);args=parser.parse_args()
record=Path(args.record);record.parent.mkdir(parents=True,exist_ok=True)
if args.action=='install':
    assert not record.exists()
    originals=[p.read_bytes() for p in files]
    backups=[]
    for i,(p,b) in enumerate(zip(files,originals)):
        q=record.parent/f'file{i}.orig';q.write_bytes(b);backups.append(str(q))
    changed=[patch_proposer(originals[0].decode()).encode(),patch_handoff(originals[1].decode()).encode(),patch_serving(originals[2].decode()).encode()]
    for p,b in zip(files,changed):p.write_bytes(b)
    record.write_text(json.dumps({'paths':[str(p) for p in files],'backups':backups,
        'original_sha256':[sha(b) for b in originals],'patched_sha256':[sha(b) for b in changed]},indent=2)+'\n')
else:
    assert record.exists();info=json.loads(record.read_text())
    for p,q,expected in zip(files,info['backups'],info['original_sha256']):
        b=Path(q).read_bytes();assert sha(b)==expected;p.write_bytes(b)
    info['restored_sha256']=[sha(p.read_bytes()) for p in files]
    record.write_text(json.dumps(info,indent=2)+'\n')
print(record)
