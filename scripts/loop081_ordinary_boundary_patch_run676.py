"""Prepare only the missing cache-ON ordinary Product boundaries, no device events.

Functions accept a reviewed selected Current source. No installation on import.
"""
def one(src,old,new):
    if src.count(old)!=1:raise RuntimeError(f'anchor count {src.count(old)}: {old[:100]}')
    return src.replace(old,new,1)

def core(src):
    src=one(src,'    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:',
        '    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:')
    src=one(src,'                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)',
        '''                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        from diagnostics.ordinary_boundary.observer import core_event
                        core_event('socket_add_decoded',req_id=req.request_id,external_req_id=req.external_req_id)''')
    src=one(src,'        req = Request.from_engine_core_request(request, self.request_block_hasher)',
        '''        from diagnostics.ordinary_boundary.observer import core_event
        core_event('preprocess_begin',req_id=request.request_id)
        req = Request.from_engine_core_request(request, self.request_block_hasher)''')
    src=one(src,'        return req, request.current_wave',
        "        core_event('preprocess_end',req_id=req.request_id)\n        return req, request.current_wave")
    src=one(src,'                    self.input_queue.put_nowait((request_type, request))',
        '''                    if request_type == EngineCoreRequestType.ADD:
                        from diagnostics.ordinary_boundary.observer import core_event
                        core_event('queue_put_begin',req_id=request[0].request_id)
                    self.input_queue.put_nowait((request_type, request))
                    if request_type == EngineCoreRequestType.ADD:
                        core_event('queue_put_end',req_id=request[0].request_id)''')
    src=one(src,'    def _process_engine_step(self) -> bool:',
        '''    def _process_engine_step(self) -> bool:''')
    src=one(src,'        # Step the engine core.\n        outputs, model_executed = self.step_fn()',
        '''        from diagnostics.ordinary_boundary.observer import core_event
        core_event('input_drain_complete')
        # Step the engine core.
        outputs, model_executed = self.step_fn()''')
    # Both synchronous and asynchronous schedules retain original behavior.
    anchor='scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())'
    assert src.count(anchor)==2
    lines=[]
    for line in src.splitlines(True):
        if anchor not in line:lines.append(line);continue
        indent=line[:len(line)-len(line.lstrip())]
        block="""from diagnostics.ordinary_boundary.observer import enabled as _ob_enabled, core_event as _ob_core
_ob_active = _ob_enabled()
if _ob_active: _ob_core('schedule_begin')
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
if _ob_active:
    self._ob_submit_seq = getattr(self, '_ob_submit_seq', 0) + 1
    scheduler_output._ob_submit_id = self._ob_submit_seq
    _ob_core('schedule_end_submit', submit_id=self._ob_submit_seq, scheduled=dict(scheduler_output.num_scheduled_tokens))
"""
        lines.extend(indent+x+'\n' for x in block.splitlines())
    src=''.join(lines)
    anchor='        self._attach_iteration_details(engine_core_outputs, iteration_details)'
    assert src.count(anchor)==2
    src=src.replace(anchor,"""        if getattr(scheduler_output, '_ob_submit_id', None) is not None:
            from diagnostics.ordinary_boundary.observer import core_event
            core_event('scheduler_settled',submit_id=scheduler_output._ob_submit_id,
                       bulk=bool(model_output.extreme_bulk_output),
                       outputs=[{'req_id':r.request_id,'count':len(r.new_token_ids),'finished':r.finished}
                                for packet in engine_core_outputs.values() for r in packet.outputs])
"""+anchor)
    return src

def runner(src):
    entry='''        from diagnostics.ordinary_boundary.observer import enabled as _ob_enabled, worker_event as _ob_event
        self._ob_current_submit = getattr(scheduler_output, '_ob_submit_id', None)
        _ob_active = _ob_enabled() and self._ob_current_submit is not None
        if _ob_active:
            _ob_event(self,'execute_entry',submit_id=self._ob_current_submit,
                total_scheduled_tokens=int(scheduler_output.total_num_scheduled_tokens),
                new=[{'req_id':x.req_id,'prompt_len':len(x.prompt_token_ids) if x.prompt_token_ids is not None else None,
                      'computed':int(x.num_computed_tokens),'scheduled':int(scheduler_output.num_scheduled_tokens.get(x.req_id,0)),'spec_tokens':len(scheduler_output.scheduled_spec_decode_tokens.get(x.req_id,[]))}
                     for x in scheduler_output.scheduled_new_reqs],
                cached=[{'req_id':rid,'computed':int(c),'output_positions_including_placeholders':int(o),
                         'scheduled':int(scheduler_output.num_scheduled_tokens.get(rid,0)),'spec_tokens':len(scheduler_output.scheduled_spec_decode_tokens.get(rid,[]))}
                        for rid,c,o in zip(scheduler_output.scheduled_cached_reqs.req_ids,
                            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                            scheduler_output.scheduled_cached_reqs.num_output_tokens)])
'''
    src=one(src,'    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if self.vllm_config.model_config.enable_return_routed_experts:',
        '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n'+entry+'        if self.vllm_config.model_config.enable_return_routed_experts:')
    src=one(src,'            self._extreme_runtime_started = True\n',
        "            if _ob_active:_ob_event(self,'handoff_entry',submit_id=self._ob_current_submit,req_ids=list(self.input_batch.req_ids))\n            self._extreme_runtime_started = True\n")
    src=one(src,'            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )',
        '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n'+
        "            if _ob_active:_ob_event(self,'runtime_built',submit_id=self._ob_current_submit,req_ids=list(_extreme_req_ids))")
    src=one(src,'                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(',
        "                if _ob_active:_ob_event(self,'runtime_begin',submit_id=self._ob_current_submit,req_ids=list(_extreme_req_ids))\n                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(")
    src=one(src,'                torch.npu.synchronize()\n                _extreme_wall_seconds = (',
        "                torch.npu.synchronize()\n                if _ob_active:_ob_event(self,'runtime_end',submit_id=self._ob_current_submit,req_ids=list(_extreme_req_ids))\n                _extreme_wall_seconds = (")
    src=one(src,'                self._extreme_serving_output = ModelRunnerOutput(',
        "                if _ob_active:\n                    from diagnostics.ordinary_boundary.observer import flush as _ob_flush\n                    _ob_flush(self,_rank,_cohort_index)\n                self._extreme_serving_output = ModelRunnerOutput(")
    src=one(src,'            self._draft_token_ids = self.propose_draft_token_ids(\n',
        '''            from diagnostics.ordinary_boundary.observer import enabled as _ob_enabled, worker_event as _ob_event
            _ob_prop_active = _ob_enabled() and getattr(scheduler_output,'_ob_submit_id',None) is not None
            if _ob_prop_active:
                _ob_event(self,'seed_begin',submit_id=getattr(scheduler_output,'_ob_submit_id',None),req_ids=list(self.input_batch.req_ids))
            self._draft_token_ids = self.propose_draft_token_ids(
''')
    src=one(src,'            self._copy_draft_token_ids_to_cpu(scheduler_output)\n',
        '''            if _ob_prop_active:
                _ob_event(self,'seed_propose_end',submit_id=getattr(scheduler_output,'_ob_submit_id',None),req_ids=list(self.input_batch.req_ids))
            self._copy_draft_token_ids_to_cpu(scheduler_output)
            if _ob_prop_active:
                _ob_event(self,'seed_end',submit_id=getattr(scheduler_output,'_ob_submit_id',None),req_ids=list(self.input_batch.req_ids))
''')
    src+='''
# Host-only ordinary forward context observer; fixed-serving target graph is untouched.
if os.getenv('EXTREME_ORDINARY_BOUNDARY_DIR'):
    _ob_original_forward=NPUModelRunner._model_forward
    def _ob_forward(self,num_tokens_padded,*args,**kwargs):
        from diagnostics.ordinary_boundary.observer import enabled,worker_event,metadata_scalars
        if not enabled() or getattr(self,'_ob_current_submit',None) is None:return _ob_original_forward(self,num_tokens_padded,*args,**kwargs)
        ctx=get_forward_context()
        worker_event(self,'target_begin',submit_id=getattr(self,'_ob_current_submit',None),req_ids=list(self.input_batch.req_ids),num_tokens_padded=int(num_tokens_padded),selected_mode=ctx.cudagraph_runtime_mode.name,attention=metadata_scalars(ctx.attn_metadata))
        try:return _ob_original_forward(self,num_tokens_padded,*args,**kwargs)
        finally:worker_event(self,'target_end',submit_id=getattr(self,'_ob_current_submit',None),req_ids=list(self.input_batch.req_ids))
    NPUModelRunner._model_forward=_ob_forward
'''
    return src
