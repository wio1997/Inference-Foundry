"""Host-only counts at actual native target forward dispatch; no tensor reads/waits."""
import functools
import json
import os
from pathlib import Path
import time


def install(ns):
    Runner = ns['NPUModelRunner']
    root = Path('/out/adopt-trace'); root.mkdir(exist_ok=True)
    original_execute, original_forward = Runner.execute_model, Runner._model_forward
    @functools.wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        self._glm53_schedule = scheduler_output
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            self._glm53_schedule = None
    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        schedule = getattr(self, '_glm53_schedule', None)
        rows = []
        if schedule is not None and schedule.total_num_scheduled_tokens and not (root / 'disabled').exists():
            for i, rid in enumerate(self.input_batch.req_ids):
                if rid not in schedule.num_scheduled_tokens:
                    continue
                request = self.requests[rid]
                pos = int(self.input_batch.num_computed_tokens_cpu[i])
                rows.append({'req_id': rid, 'computed_before': pos,
                    'scheduled_tokens': schedule.num_scheduled_tokens[rid],
                    'host_query_token': int(self.input_batch.token_ids_cpu[i, pos]),
                    'output_history_count': len(request.output_token_ids),
                    'initial_history_head': list(request.output_token_ids[:2])})
        result = original_forward(self, *args, **kwargs)
        if rows:
            record = {'kind': 'target_forward_dispatch_returned', 'monotonic_ns': time.monotonic_ns(),
                'pid': os.getpid(), 'tp_rank': self.tp_rank, 'requests': rows,
                'scope': 'Actual native target _model_forward dispatch count; host input-batch state, not device input readback, kernel count or device duration'}
            with (root / ('worker-%s.jsonl' % os.getpid())).open('a') as f:
                f.write(json.dumps(record, sort_keys=True) + '\n')
        return result
    Runner.execute_model, Runner._model_forward = execute, forward
