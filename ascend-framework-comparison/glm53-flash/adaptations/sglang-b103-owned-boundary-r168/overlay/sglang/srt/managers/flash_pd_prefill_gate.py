# SPDX-License-Identifier: Apache-2.0
"""Test-only readiness gate; native scheduler still chooses the real batch."""
import os,time

def should_hold(scheduler):
 if os.environ.get('FLASH_PD_PREFILL_COHORT_GATE')!='1':return False
 from sglang.srt.managers.flash_pd_request_context import enabled,_book
 if not enabled() or not scheduler.waiting_queue:return False
 from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
 import torch
 book=_book();actual=[];groups=set()
 for req in scheduler.waiting_queue:
  tag=book['by_id'][req.rid]
  if len(req.origin_input_ids)!=tag['input_length'] or int(req.bootstrap_room)!=tag['bootstrap_room']:raise ValueError('Wrong actual native Prefill readiness identity')
  actual.append((req,tag));groups.add((tag['repeat'],tag['cohort']))
 if len(groups)!=1:raise ValueError('Test readiness gate received mixed cohorts')
 key=next(iter(groups));released=getattr(scheduler,'_flash_pd_ready_cohorts',None)
 if released is None:released=scheduler._flash_pd_ready_cohorts=set()
 if key in released:return False
 if scheduler.chunked_req is not None:raise ValueError('Unreleased cohort already started model computation')
 expected=[t for t in book['by_id'].values() if (t['repeat'],t['cohort'])==key]
 if len(expected)!=2:raise ValueError('Readiness gate expects exactly two owned requests')
 ready=[tag for req,tag in actual if not req.pending_bootstrap]
 ready_rids={t['rid'] for t in ready};required={t['rid'] for t in expected}
 starts=getattr(scheduler,'_flash_pd_cohort_wait_started',None)
 if starts is None:starts=scheduler._flash_pd_cohort_wait_started={}
 if ready_rids==required:
  released.add(key);append('prefill_cohort_ready',device=torch.npu.current_device(),role=os.environ['FLASH_PD_ROLE'],requests=ready,req_available=int(scheduler.req_to_token_pool.available_size()),mamba_available=int(scheduler.req_to_token_pool.mamba_allocator.available_size()),test_only_gate=True);return False
 if key not in starts:
  starts[key]=time.monotonic();append('prefill_cohort_waiting',device=torch.npu.current_device(),role=os.environ['FLASH_PD_ROLE'],requests=ready,required_rids=sorted(required),test_only_gate=True)
 if time.monotonic()-starts[key]>30:raise TimeoutError('Owned Prefill cohort did not actually bootstrap within30s')
 return True
