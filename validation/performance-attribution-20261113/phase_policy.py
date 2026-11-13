"""固定vLLM调度器的批次契约干预；仅主机元数据，不计算模型。"""
from collections import Counter
from functools import lru_cache
from numbers import Integral
from pathlib import Path
import ast
import hashlib
import json
import textwrap
import time

CORE_SHA256='c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884'


def integer(value):
    if isinstance(value,bool) or not isinstance(value,Integral):
        raise TypeError('expected host integer;no tensor conversion allowed')
    return int(value)


@lru_cache(maxsize=64)
def identity(rid):
    if not isinstance(rid,str) or not rid or len(rid)>256:raise ValueError('invalid host request ID')
    return hashlib.sha256(rid.encode()).hexdigest()


class PhasePolicy:
    def __init__(self,enabled,source_sha,stats_path,*,runtime_identity=None):
        if type(enabled) is not bool:raise TypeError('explicit host policy flag required')
        self.enabled=enabled;self.source_sha=source_sha;self.stats_path=Path(stats_path)
        self.runtime_identity=dict(runtime_identity or {})
        self.policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.counts=Counter();self.events=[];self.closed=False
        self.clock={'boot_id_sha256':hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),
                    'timens_offsets_sha256':hashlib.sha256(Path('/proc/self/timens_offsets').read_bytes()).hexdigest(),
                    'monotonic_implementation':time.get_clock_info('monotonic').implementation,
                    'perf_counter_implementation':time.get_clock_info('perf_counter').implementation}

    def begin(self):
        if self.closed:raise RuntimeError('closed scheduler policy used')
        self.counts['schedule_calls']+=1
        return time.monotonic_ns()

    def phase(self,request,computed,scheduled,pad_spec_decode=False):
        # Match the pinned scheduler's actual construction of spec-token rows.
        computed=integer(computed);scheduled=integer(scheduled)
        if computed<0 or scheduled<1:raise ValueError('invalid scheduled host range')
        if request.has_encoder_inputs or integer(request.num_preemptions):
            raise ValueError('fixed text experiment does not cover encoders or preemption')
        spec=request.spec_token_ids
        if not isinstance(spec,(list,tuple)):raise TypeError('spec IDs must be host sequence')
        spec_rows=scheduled+computed-integer(request.num_tokens)-integer(request.num_output_placeholders)
        if pad_spec_decode or (spec and spec_rows>0):return 'spec_decode'
        if type(request.is_prefill_chunk) is not bool:raise TypeError('prefill marker must be host boolean')
        if request.is_prefill_chunk or computed<integer(request.num_prompt_tokens):return 'prefill'
        return 'decode'

    def allows(self,selected,candidate):
        allowed=not self.enabled or selected is None or selected==candidate
        if not allowed:self.counts['phase_deferrals']+=1
        return allowed

    def record(self,scheduler,output,tags,started):
        if output.preempted_req_ids:raise ValueError('preempted run is outside this fixed diagnostic')
        keys=set(output.num_scheduled_tokens)
        if not keys<=set(tags):raise RuntimeError('unobserved allocation in scheduler output')
        phases={tags[r] for r in keys}
        if self.enabled and len(phases)>1:raise RuntimeError('phase policy produced mixed contracts')
        rows=[]
        for rid in output.num_scheduled_tokens:
            request=scheduler.requests[rid];spec=output.scheduled_spec_decode_tokens.get(rid,())
            if not isinstance(spec,(list,tuple)):raise TypeError('scheduled spec IDs must be host sequence')
            if (tags[rid]=='spec_decode')!=bool(spec):raise RuntimeError('policy class differs from scheduled spec rows')
            rows.append({'request_id_sha256':identity(rid),'phase':tags[rid],
                'computed':integer(request.num_computed_tokens),'prompt':integer(request.num_prompt_tokens),
                'scheduled':integer(output.num_scheduled_tokens[rid]),'spec_tokens':len(spec)})
        total=sum(r['scheduled'] for r in rows)
        if total!=integer(output.total_num_scheduled_tokens):raise RuntimeError('scheduled row total differs')
        if not keys:
            self.counts['empty_steps']+=1
            return
        self.counts['nonempty_steps']+=1
        if len(phases)>1:self.counts['mixed_contract_steps']+=1
        for phase in phases:self.counts['steps_with_'+phase]+=1
        if len(self.events)<2048:
            self.events.append({'started_monotonic_ns':started,'returned_metadata_monotonic_ns':time.monotonic_ns(),
                                'rows':rows,'total_scheduled_tokens':total})
        else:self.counts['events_dropped']+=1

    def close(self,shutdown_error=None):
        if self.closed:return
        self.closed=True
        value={'closed':True,'enabled':self.enabled,'shutdown_error':shutdown_error,
            'source_sha256':self.source_sha,'counts':dict(self.counts),'events':self.events,'clock':self.clock,
            'policy_source_sha256':self.policy_source_sha256,'runtime_identity':self.runtime_identity,
            'scope':'scheduler host metadata before update_after_schedule;not worker graph mode or device time',
            'new_model_math':False}
        tmp=self.stats_path.with_suffix('.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(self.stats_path)


def transformed_method(module_source):
    if hashlib.sha256(module_source.encode()).hexdigest()!=CORE_SHA256:
        raise ValueError('fixed SDK scheduler source identity changed')
    tree=ast.parse(module_source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Scheduler')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='schedule')
    code=textwrap.dedent('\n'.join(module_source.splitlines()[method.lineno-1:method.end_lineno]))+'\n'
    replacements=[
        ('    self.current_step += 1\n','    self.current_step += 1\n    _ap=self._attribution_phase_policy\n    _ap_started=_ap.begin()\n    _ap_phase=None\n    _ap_tags={}\n'),
        ('        # Schedule newly needed KV blocks for the request.\n',
         '        _ap_candidate=_ap.phase(request,request.num_computed_tokens,num_new_tokens)\n'
         '        if not _ap.allows(_ap_phase,_ap_candidate):\n            req_index += 1\n            continue\n'
         '        # Schedule newly needed KV blocks for the request.\n'),
        ('        scheduled_running_reqs.append(request)\n',
         '        scheduled_running_reqs.append(request)\n        if _ap_phase is None: _ap_phase=_ap_candidate\n        _ap_tags[request.request_id]=_ap_candidate\n'),
        ('            reserved_blocks = 0\n',
         '            _ap_candidate=_ap.phase(request,num_computed_tokens,num_new_tokens,pad_spec_decode)\n'
         '            if not _ap.allows(_ap_phase,_ap_candidate):\n                break\n'
         '            reserved_blocks = 0\n'),
        ('            self.running.append(request)\n',
         '            self.running.append(request)\n            if _ap_phase is None: _ap_phase=_ap_candidate\n            _ap_tags[request.request_id]=_ap_candidate\n'),
        ('    with record_function_or_nullcontext("schedule: update_after_schedule"):\n',
         '    _ap.record(self,scheduler_output,_ap_tags,_ap_started)\n'
         '    with record_function_or_nullcontext("schedule: update_after_schedule"):\n'),
    ]
    for before,after in replacements:
        if code.count(before)!=1:raise ValueError('fixed scheduling anchor changed: '+before.strip())
        code=code.replace(before,after,1)
    compile(code,'<fixed-sdk-phase-policy>','exec')
    return code
