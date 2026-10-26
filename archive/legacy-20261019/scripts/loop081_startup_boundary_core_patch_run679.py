"""Off-path Core patch: buffered request/submit joins and publication boundaries."""
from loop081_ordinary_boundary_patch_run676 import core as old_core, one

def core(src):
    src=old_core(src).replace('diagnostics.ordinary_boundary.observer','diagnostics.startup_boundary.observer')
    src=one(src,"core_event('socket_add_decoded',req_id=req.request_id,external_req_id=req.external_req_id)",
        "core_event('socket_add_decoded',req_id=req.request_id,external_id=req.external_req_id)")
    src=one(src,"        core_event('input_drain_complete')",
        "        self._ob_last_settled_submit = None\n        core_event('input_drain_complete')")
    # Both sync and async paths have resolved model_output at this point.
    anchor='        # Before processing the model output, process any aborts that happened'
    assert src.count(anchor)==2
    src=src.replace(anchor,"""        self._ob_last_settled_submit = getattr(scheduler_output, '_ob_submit_id', None)
        if self._ob_last_settled_submit is not None:
            from diagnostics.startup_boundary.observer import core_event
            core_event('result_ready',submit_id=self._ob_last_settled_submit)
"""+anchor)
    anchor='        engine_core_outputs = self.scheduler.update_from_output(\n            scheduler_output, model_output\n        )'
    assert src.count(anchor)==2
    src=src.replace(anchor,"""        if self._ob_last_settled_submit is not None:
            core_event('scheduler_update_begin',submit_id=self._ob_last_settled_submit)
"""+anchor+"""
        if self._ob_last_settled_submit is not None:
            core_event('scheduler_update_end',submit_id=self._ob_last_settled_submit)
""")
    anchor='        for output in outputs.items() if outputs else ():\n            self.output_queue.put_nowait(output)\n'
    src=one(src,anchor,"""        from diagnostics.startup_boundary.observer import core_event, core_flush
        for output in outputs.items() if outputs else ():
            _ob_id = getattr(self, '_ob_last_settled_submit', None)
            if _ob_id is not None:
                core_event('output_enqueue_begin',submit_id=_ob_id,client_index=output[0],req_ids=[r.request_id for r in output[1].outputs])
            self.output_queue.put_nowait(output)
            if _ob_id is not None:
                core_event('output_enqueue_end',submit_id=_ob_id,client_index=output[0])
        if outputs:
            core_flush()
""")
    return src
