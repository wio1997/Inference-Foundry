# SPDX-License-Identifier: Apache-2.0
"""SHA-pinned draft native lifetime diagnostic hooks; no model math changes."""
import argparse,ast,pathlib,hashlib,runpy

def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Wrong native lifetime hook anchor: '+a[:100])
 return s.replace(a,b,1)

def add_import(s,line):
 tree=ast.parse(s);offset=0
 for i,n in enumerate(tree.body):
  if (i==0 and isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or (isinstance(n,ast.ImportFrom) and n.module=='__future__'):offset=n.end_lineno
  else:
   if offset==0:offset=n.lineno-1
   break
 lines=s.splitlines(keepends=True);lines.insert(offset,line+'\n');return ''.join(lines)

def decode(s):
 patch=runpy.run_path(str(pathlib.Path(__file__).with_name('prepare_b103_pd_drain_release.py')))['patch'];s=patch(s)
 s=add_import(s,'from sglang.srt.managers.flash_pd_lifetime_audit import deferred as _flash_lifetime_deferred, resolve as _flash_lifetime_resolve, release as _flash_lifetime_release, metadata_before_publish as _flash_metadata_before_publish, intake as _flash_lifetime_intake, admission_blocked as _flash_admission_blocked, drain_fence as _flash_drain_fence')
 s=once(s,'        if drain_ready.numel() and torch.distributed.get_world_size(self.gloo_group) > 1:\n','        _flash_local_drain_ready = drain_ready.tolist()\n        if drain_ready.numel() and torch.distributed.get_world_size(self.gloo_group) > 1:\n')
 s=once(s,'        drain_ready = iter(drain_ready.tolist())\n','        _flash_drain_fence(self, _flash_local_drain_ready, drain_ready.tolist())\n        drain_ready = iter(drain_ready.tolist())\n')
 for name,decorator in [('_defer_release','deferred'),('_do_release','release'),('resolve_deferred_releases','resolve')]:
  anchor='    def '+name+'(';s=once(s,anchor,'    @_flash_lifetime_'+decorator+'\n'+anchor)
 anchor='    def add(\n        self, req: Req, is_retracted: bool = False, is_rebootstrap: bool = False\n';s=once(s,anchor,'    @_flash_lifetime_intake\n'+anchor)
 anchor='            if self.req_to_token_pool.available_size() <= 0:\n                break\n\n            # Hybrid models';s=once(s,anchor,'            if self.req_to_token_pool.available_size() <= 0:\n                _flash_admission_blocked(self, decode_req)\n                break\n\n            # Hybrid models')
 anchor='                if mamba_allocator.available_size() <= 0:\n                    break\n';s=once(s,anchor,'                if mamba_allocator.available_size() <= 0:\n                    _flash_admission_blocked(self, decode_req)\n                    break\n')
 anchor='        decode_req.kv_receiver.send_metadata(\n';s=once(s,anchor,'        _flash_metadata_before_publish(self, decode_req, page_indices, state_indices)\n'+anchor);return s

def common(s):
 # Arm before publishing ABORT: an immediate real sender ACK must have a live
 # tracker, and a new receiver must clear a previous owner's stale room ACKs.
 anchor='    def _send_abort_notification(self):\n'
 s=once(s,anchor,anchor+'''        if self.kv_mgr.enable_deferred_decode_kv_release:
            self.kv_mgr.register_deferred_abort_room(self.bootstrap_room)
''')
 s=add_import(s,'from sglang.srt.managers.flash_pd_lifetime_audit import common_action as _flash_lifetime_common')
 for name in ['register_deferred_abort_room','note_abort_ack','_send_abort_ack','register_deferred_ack_target']:
  anchor='    def '+name+'(';s=once(s,anchor,'    @_flash_lifetime_common\n'+anchor)
 return s

def inventory(s):
 anchor="    append('native_inventory',device=e.npu_id,role=e.role,session_id=e.session_id,"
 s=once(s,anchor,"    from sglang.srt.managers.flash_pd_lifetime_audit import inventory as _lifetime_inventory\n    _lifetime_inventory(e.npu_id, rows)\n"+anchor);return s

def engine(s):
 anchor="            _event('push_submitted', device=self.npu_id, remote_session=str(session_id),"
 s=once(s,anchor,"            from sglang.srt.managers.flash_pd_lifetime_audit import before_actual_write\n            before_actual_write(self, buffers, lengths, active)\n"+anchor);return s

def scheduler(s):
 # CommonKVReceiver._send_abort_notification now arms BEFORE publishing ABORT.
 # Re-arming here would erase an immediate real ACK, or a prior ACK on a
 # repeated abort of the same receiver. Its existing abort_notified guard owns
 # exactly-once arming/publication; keep that lifecycle in one place.
 old='''                    # Arm drain-ack accounting once the ABORT is sent, so acks
                    # arriving before this req is deferred (e.g. during the next
                    # forward step) are captured. A fresh set also drops stale acks
                    # from a prior request that reused this bootstrap_room. A
                    # redundant abort only re-wipes -- holds longer, never releases
                    # early -- so no transition guard is needed.
                    if (
                        receiver.kv_mgr.enable_deferred_decode_kv_release
                        and receiver.abort_notified
                    ):
                        receiver.kv_mgr.register_deferred_abort_room(
                            decode_req.req.bootstrap_room
                        )
'''
 return once(s,old,'                    # Receiver arms once before its actual ABORT publication.\n')

def sender(s):
 old='''            # Hold Success until all staging chunks transferred: a deferred
            # chunk can still be pending, and concluding now would drop it.
            if (
                status == KVPoll.Success
                and self.kv_mgr._staging_outstanding.get(self.bootstrap_room, 0) > 0
            ):
                return KVPoll.Transferring
'''
 new='''            # Both terminal states must retain source buffers while a real
            # dequeued SDK write can still read them, including an aborted write.
            _flash_pending_writes = self.kv_mgr._staging_outstanding.get(self.bootstrap_room, 0)
            if status in (KVPoll.Success, KVPoll.Failed) and _flash_pending_writes > 0:
                from sglang.srt.managers.flash_pd_lifetime_audit import sender_source_hold as _flash_sender_source_hold
                _flash_sender_source_hold(self, status, _flash_pending_writes)
                return KVPoll.Transferring
'''
 return once(s,old,new)

def prefill(s):
 # Native poll consensus propagates Failed immediately by MIN. A rank whose
 # SDK work already drained must not cause other ranks to free live sources.
 old='''        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
'''
 new='''        from sglang.srt.managers.flash_pd_lifetime_audit import prefill_source_fence as _flash_source_fence
        polls = _flash_source_fence(self, polls)

'''+old
 return once(s,old,new)

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['decode','common','inventory','engine','scheduler','sender','prefill'],required=True);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();b=a.source.read_bytes()
 if hashlib.sha256(b).hexdigest()!=a.expected_sha256:raise ValueError('Wrong pinned lifetime source')
 s=globals()[a.kind](b.decode());ast.parse(s);a.output.write_text(s)
if __name__=='__main__':main()
