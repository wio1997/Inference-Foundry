"""Off-path patch functions; no mutation of live source on import."""
def one(src,old,new):
 if src.count(old)!=1:raise RuntimeError(f'anchor count {src.count(old)}: {old[:80]}')
 return src.replace(old,new,1)

def handoff(src):
 src=one(src,"        self._defer_slot_audit = self._slot_audit_mode['enabled']\n        self._pending_slot_refresh_audit = []",
 """        self._defer_slot_audit = self._slot_audit_mode['enabled']
        self._pending_slot_refresh_audit = []
        self._startup_slot_certificate = None
        self._slot_certificate_mode = read_cohort_mode(os.environ['EXTREME_SLOT_CERTIFICATE_MODE_FILE'])""")
 anchor='    def _flush_slot_refresh_audit(self) -> None:'
 src=one(src,anchor,'''    def certify_startup_slots(self, state, positions, remaining, schedule_mode) -> None:
        if not self._slot_certificate_mode['enabled']:
            return
        from runtime.startup_slot_certificate import StartupSlotCertificate
        self._startup_slot_certificate = StartupSlotCertificate.build(
            bindings=self._group_slot_bindings, positions=list(positions),
            remaining=list(remaining), cycle_index=state.cycle_index,
            schedule_mode=schedule_mode, batch_size=self.config.batch_size,
            width=self.config.target_tokens_per_request, computed_dtype=state.num_computed_tokens.dtype)

'''+anchor)
 src=one(src,'        for gid, table, mapping, block_size in self._group_slot_bindings:\n            logical =',
 '''        guard_required = state.cycle_index < 8
        if guard_required and self._startup_slot_certificate is not None:
            guard_required = not self._startup_slot_certificate.permits(
                state.cycle_index, self._group_slot_bindings)
        for gid, table, mapping, block_size in self._group_slot_bindings:
            logical =''')
 src=one(src,'            if state.cycle_index < 8 and bool((logical < 0).any() or (logical >= table.shape[1]).any()):',
 '            if guard_required and bool((logical < 0).any() or (logical >= table.shape[1]).any()):')
 src=one(src,'            if state.cycle_index < 8 and bool((blocks < 0).any()):',
 '            if guard_required and bool((blocks < 0).any()):')
 return src

def serving(src):
 return one(src,'''        self._initial_positions = [
            int(value) for value in runtime.state.num_computed_tokens.cpu().tolist()
        ]''','''        self._initial_positions = [
            int(value) for value in runtime.state.num_computed_tokens.cpu().tolist()
        ]
        certify = getattr(runtime.proposer, 'certify_startup_slots', None)
        if certify is not None:
            certify(runtime.state, self._initial_positions, self.remaining, runtime._schedule_mode)''')

def runner(src):
 return one(src,'                    "slot_audit_pending": len(_extreme_runtime.proposer._pending_slot_refresh_audit),',
 '''                    "slot_audit_pending": len(_extreme_runtime.proposer._pending_slot_refresh_audit),
                    "slot_certificate_mode": dict(_extreme_runtime.proposer._slot_certificate_mode),
                    "slot_certificate": (dict(_extreme_runtime.proposer._startup_slot_certificate.record)
                        if _extreme_runtime.proposer._startup_slot_certificate is not None else None),''')
