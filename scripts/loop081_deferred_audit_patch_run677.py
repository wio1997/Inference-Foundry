"""Defer diagnostic scalar extraction only; retain all immediate safety guards."""
def one(src,old,new):
 if src.count(old)!=1:raise RuntimeError(f'anchor count {src.count(old)}: {old[:80]}')
 return src.replace(old,new,1)

def patch(src):
 src=one(src,'        self.slot_refresh_audit: list[dict[str, object]] = []',
 '''        self.slot_refresh_audit: list[dict[str, object]] = []
        from serving.cohort_mode import read_cohort_mode
        self._slot_audit_mode = read_cohort_mode(os.environ['EXTREME_SLOT_AUDIT_MODE_FILE'])
        self._defer_slot_audit = self._slot_audit_mode['enabled']
        self._pending_slot_refresh_audit = []''')
 src=one(src,'        self._commit_host_mirrors()\n        batch = self.config.batch_size\n        width = self.config.target_tokens_per_request\n        query = state.target_query_start_loc.cpu()',
 '''        self._flush_slot_refresh_audit()
        self._commit_host_mirrors()
        batch = self.config.batch_size
        width = self.config.target_tokens_per_request
        query = state.target_query_start_loc.cpu()''')
 anchor='    @torch.inference_mode()\n    def _refresh_draft_context_slots(self, state: FixedDecodeState) -> None:'
 src=one(src,anchor,'''    def _flush_slot_refresh_audit(self) -> None:
        pending = self._pending_slot_refresh_audit
        if not pending:
            return
        # Results were computed before each mapping mutation; they are scalars,
        # not views of mapping/table/state. Caller already ended Runtime sync.
        values = torch.stack([
            torch.stack((changed, zero)) for _, _, changed, zero in pending
        ]).cpu().tolist()
        rows = [dict(cycle=cycle, gid=gid, changed=int(value[0]), zero_blocks=int(value[1]))
                for (cycle, gid, _, _), value in zip(pending, values)]
        self.slot_refresh_audit.extend(rows)
        self._pending_slot_refresh_audit = []

'''+anchor)
 old='''            changed = int((mapping[:96].to(torch.int64) != expected).sum().item()) if state.cycle_index < 8 else None
            mapping[:96].copy_(expected.to(mapping.dtype))
            if changed is not None:
                self.slot_refresh_audit.append({"cycle": state.cycle_index, "gid": gid,
                                                "changed": changed,
                                                "zero_blocks": int((blocks == 0).sum().item())})'''
 new='''            deferred = self._defer_slot_audit and state.cycle_index < 8
            if deferred:
                # Snapshot the count before copy_, preserving original semantics.
                changed_tensor = (mapping[:96].to(torch.int64) != expected).sum()
                changed = None
            else:
                changed = int((mapping[:96].to(torch.int64) != expected).sum().item()) if state.cycle_index < 8 else None
            mapping[:96].copy_(expected.to(mapping.dtype))
            if deferred:
                zero_tensor = (blocks == 0).sum()
                if len(self._pending_slot_refresh_audit) >= 8 * len(self._group_slot_bindings):
                    raise RuntimeError('slot audit exceeded bounded startup rows')
                self._pending_slot_refresh_audit.append((state.cycle_index, gid, changed_tensor, zero_tensor))
            elif changed is not None:
                self.slot_refresh_audit.append({"cycle": state.cycle_index, "gid": gid,
                                                "changed": changed,
                                                "zero_blocks": int((blocks == 0).sum().item())})'''
 return one(src,old,new)
