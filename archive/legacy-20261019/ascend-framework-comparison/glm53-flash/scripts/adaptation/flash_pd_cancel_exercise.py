# SPDX-License-Identifier: Apache-2.0
"""Draft bounded cancellation exercise, called only by a reviewed run launcher.

Callbacks perform the frozen owned remote calls. This file does nothing when
imported. No timing/performance or cancellation result is inferred from HTTP200.
The calling launcher must stop and prove the Prefill writer stopped/idle before
stopping Decode on error; releasing the test gate alone does not prove a drain.
"""
import concurrent.futures,json,pathlib,time

def exercise(R,host_futures,request,abort_control,lifetime_events,signal,out):
 commands=json.loads((R/'commands.json').read_text());assert len(commands)==6 and [v['sequence'] for v in commands]==list(range(6))
 cancel=commands[0];fresh=commands[1];assert cancel['purpose']=='late_write_cancel' and all(v['purpose']=='fresh_normal' for v in commands[1:])
 api=concurrent.futures.ThreadPoolExecutor(max_workers=4)
 def tag(e):return e.get('tag',e.get('ownership',{}).get('tag'))
 def wait_events(node,kind,command,bound,after_first_bound=None):
  deadline=time.monotonic()+bound;first=None
  while time.monotonic()<deadline:
   assert all(not f.done() for f in host_futures.values()),'OWN_SERVICE_EXITED_DURING_CANCEL_UNIT'
   rows=[e for e in lifetime_events(node) if e['kind']==kind and tag(e)==command]
   devices={e['device'] for e in rows}
   if rows and first is None:first=time.monotonic()
   if devices==set(range(16)):
    assert len(rows)==16,'Duplicate native lifetime transition';return rows
   if first is not None and after_first_bound is not None and time.monotonic()-first>after_first_bound:break
   time.sleep(.5)
  raise TimeoutError('Missing actual native transition '+kind+' for '+command['rid'])
 def progress(stage):
  p=R/'progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'project_id':'glm53-flash','stage':stage,'completed_normal_pairs':len(out['pairs']),'formal_requests':0}));tmp.replace(p)
 try:
  dc=api.submit(request,47,cancel);pc=api.submit(request,46,cancel);progress('cancel_pair_submitted_waiting_real_KDA_packet')
  gates=wait_events(46,'late_write_gate_entered',cancel,420,after_first_bound=15)
  control=abort_control(cancel);assert control['attempted'] and control['status']==200 and not control.get('error');out['abort_control']=control
  deferred=wait_events(47,'destination_deferred',cancel,15)
  # This actual D intake must arrive while the old destination remains owned.
  df=api.submit(request,47,fresh);pf=api.submit(request,46,fresh)
  intake=wait_events(47,'decode_request_intake',fresh,15)
  for e in intake:
   assert cancel['bootstrap_room'] in e['held_rooms'] and e['req_slot'] is None and e['mamba_slot'] is None and e['mamba_available']==0
  seen_abort=wait_events(46,'native_register_deferred_ack_target_after',cancel,15)
  assert all(e['pending_writes']>0 and e['room_status'] in ('Failed',None) for e in seen_abort)
  source_held=wait_events(46,'prefill_failed_source_retained',cancel,15)
  assert all(e['pending_writes']>0 and e['poll_result']=='Transferring' and e['room_status']=='Failed' for e in source_held)
  blocked=wait_events(47,'decode_admission_blocked_held_mamba',fresh,15)
  assert all(cancel['bootstrap_room'] in e['held_rooms'] and e['waiting_for_input'] and e['mamba_available']==0 for e in blocked)
  expired=wait_events(47,'deadline_expired_destination_still_owned',cancel,15)
  assert not df.done(),'Fresh D request terminated before old owner drained'
  early=[e for e in lifetime_events(47) if e['kind']=='destination_allocated_before_publish' and tag(e)==fresh];assert not early,'Reused destination before SDK completion/drain'
  out['held_checkpoints']={'actual_P_gate_entries':gates,'actual_D_deferrals':deferred,'actual_fresh_D_intake_while_held':intake,'actual_D_expired_held':expired,'actual_P_abort_observed_while_write_pending':seen_abort,'actual_P_failed_source_held':source_held,'actual_fresh_D_ready_but_blocked_on_held_mamba':blocked}
  signal(46,'late-write-release.json',{'project_id':'glm53-flash','bootstrap_room':cancel['bootstrap_room'],'release_actual_SDMA':True});progress('actual_SDMA_unblocked_waiting_native_ACK_fence')
  released=wait_events(47,'destination_release_after',cancel,45)
  out['cancel']={'command':cancel,'prefill':pc.result(timeout=30),'decode':dc.result(timeout=30),'actual_D_safe_release_events':released}
  # Both fresh callers were genuinely queued while held. Native P has explicit
  # optimistic_prefill_attempts=0 and waits for D's safely published destination.
  out['pairs'].append({'command':fresh,'case':fresh['index'],'repeat':fresh['repeat'],'prefill':pf.result(timeout=570),'decode':df.result(timeout=570)})
  progress('first_fresh_pair_completed_after_actual_drain')
  for command in commands[2:]:
   d=api.submit(request,47,command);p=api.submit(request,46,command)
   out['pairs'].append({'command':command,'case':command['index'],'repeat':command['repeat'],'prefill':p.result(timeout=570),'decode':d.result(timeout=570)});progress('fresh_normal_pair_completed')
 finally:
  # On failure, do not leave a deliberately gated writer waiting while a caller
  # might destroy its Decode destination. This is only a real-write release
  # intent; the outer launcher's P-stopped/idle-before-D-stop fence is required.
  try:
   signal(46,'late-write-release.json',{'project_id':'glm53-flash','bootstrap_room':cancel['bootstrap_room'],'release_actual_SDMA':True})
   out['cleanup_test_gate_release_intent']=True
  except Exception as e:out['cleanup_test_gate_release_error']=repr(e)
  # Do not wait on failed callers before the launcher can signal owned cleanup.
  api.shutdown(wait=False,cancel_futures=True)
