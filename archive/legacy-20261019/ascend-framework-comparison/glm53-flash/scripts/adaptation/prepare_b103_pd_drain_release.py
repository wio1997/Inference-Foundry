# SPDX-License-Identifier: Apache-2.0
"""Candidate: retain aborted device pages until actual sender drain ACKs.

Source-only until separately qualified. A wall-clock timeout is not completion
of an outstanding SDMA write. Preserve existing host-stage behavior.
"""
import argparse,ast,pathlib,hashlib

def once(s,a,b):
    if s.count(a)!=1:raise ValueError('Wrong native drain-release anchor')
    return s.replace(a,b,1)

def patch(s):
    old='''            drained = kv_mgr.is_abort_release_safe(room, required_acks)
            if not drained and now < deadline:
                still_held.append((decode_req, deadline, idx, required_acks))
            else:
                to_release.append((decode_req, idx, room, drained))
'''
    new='''            drained = kv_mgr.is_abort_release_safe(room, required_acks)
            if not drained:
                # SDMA can still write this destination after a control-plane
                # deadline. Retain page, request, SSM and metadata ownership.
                still_held.append((decode_req, deadline, idx, required_acks))
                if now >= deadline:
                    seen = getattr(self, '_flash_sdma_expired_rooms', None)
                    if seen is None:
                        seen = self._flash_sdma_expired_rooms = set()
                    if room not in seen:
                        seen.add(room)
                        logger.error(
                            "Flash PD drain deadline expired for room %s; "
                            "retaining ownership until sender ACKs", room)
            else:
                to_release.append((decode_req, idx, room, True))
'''
    s=once(s,old,new)
    old='''    def _do_release(self, decode_req: DecodeRequest, idx: int) -> None:
        room = decode_req.req.bootstrap_room
'''
    new=old+'''        seen = getattr(self, '_flash_sdma_expired_rooms', None)
        if seen is not None:
            seen.discard(room)
'''
    s=once(s,old,new)
    # Native transfer polling already synchronizes the deferred list/order.
    # Extend its existing host-buffer readiness MIN to every held destination,
    # so device/request/SSM allocation order cannot diverge across TP ranks.
    start=s.index('    def resolve_deferred_releases(self) -> None:')
    end=s.index('    def release_memory_occupation(self):',start)
    method=s[start:end]
    method=once(method,'"""Release drained transfers; device destinations may also time out."""','"""Release held destinations only after all TP ranks observe sender drain."""')
    method=once(method,'                if self.enable_host_receive and req.host_staged\n','')
    method=method.replace('host_ready','drain_ready')
    method=once(method,'# Admission must see identical host capacity and page order on every rank.','# Admission must see identical destination capacity and page order on every rank.')
    method=once(method,'            room = decode_req.req.bootstrap_room\n','            room = decode_req.req.bootstrap_room\n            drained = bool(next(drain_ready))\n')
    method=once(method,'                if next(drain_ready):','                if drained:')
    method=once(method,'            kv_mgr = decode_req.kv_receiver.kv_mgr\n            drained = kv_mgr.is_abort_release_safe(room, required_acks)\n','')
    warning='''            if not drained:
                logger.warning(
                    f"Deferred KV release for room {room} timed out after "
                    f"{self.deferred_kv_release_timeout}s without a full drain "
                    f"ack from prefill; releasing anyway."
                )
'''
    method=once(method,warning,'            assert drained, "Flash PD release requires globally observed drain"\n')
    s=s[:start]+method+s[end:];ast.parse(s);return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();b=a.source.read_bytes()
    if hashlib.sha256(b).hexdigest()!=a.expected_sha256:raise ValueError('Wrong fixedB103 decode source')
    a.output.write_text(patch(b.decode()))
if __name__=='__main__':main()
