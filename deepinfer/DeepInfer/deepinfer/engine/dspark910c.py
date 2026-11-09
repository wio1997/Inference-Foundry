# SPDX-License-Identifier: Apache-2.0
"""Native DSpark greedy rejection sampling for the fixed DP2/TP8 service.

Candidates come exclusively from checkpoint mtp layers, never from text
lookup. All accepted outputs are checked against the full target model.
"""
from .eos_state910c import finish_reason

import torch
import torch.distributed as dist
import time
import os


def target_tokens(runner,logits,*,return_device=False,defer_host=False):
    shape=logits.shape[:-1]
    if (runner.model.target_vocab_parallel and
            logits.shape[-1]==runner.model.target_vocab_width):
        from deepinfer.npu_ops.greedy910c import global_argmax
        result=global_argmax(logits,runner.tp_group,runner.model.cfg['vocab_size'])
        runner.model.target_vocab_top1_calls+=1
    else:
        result=logits.argmax(-1).long() if runner.tp_rank==0 else torch.empty(
            shape,dtype=torch.int64,device=runner.model.device)
        dist.broadcast(result,runner.root,group=runner.tp_group)
    if defer_host:
        if return_device:raise ValueError('deferred mode already returns device tokens')
        return result
    values=result.cpu().tolist()
    return (values,result) if return_device else values


@torch.inference_mode()
def generate(runner,owned,logits,hits,emit,started):
    model=runner.model
    draft=model.dspark
    count=len(owned)
    if count!=runner.slots or count!=4:
        raise ValueError('DSpark fixed profile requires four requests per DP')
    model.use_decode_graph=False
    n=draft.nextn
    width=n+1
    model.mc2_global_bs=((count*width+runner.tp_size-1)//runner.tp_size)*dist.get_world_size()
    maximum=[r['max_tokens'] for _,r in owned]
    generated=[[] for _ in owned]
    finished=[False]*count
    eos=runner.engine.model.reader.config['eos_token_id']
    stats={'method':'dspark','max_draft_tokens':n,'draft_network_layers':len(draft.layers),
        'target_feature_layers':list(draft.target_layer_ids),'verification_query_tokens':width,
        'confidence_head_executed':False,'markov_head_executed':False,
        'verification_mode':'fixed-5','proposals':0,'draft_tokens_verified':0,
        'draft_tokens_accepted':0,'emitted_target_tokens':0,'repair_forwards':0,
        'draft_and_sampling_wall_seconds':0.,'target_and_sampling_wall_seconds':0.,
        'commit_and_emit_wall_seconds':0.}
    model.dspark_stats=stats
    shadow=None
    persistent_state_audit=None
    if os.environ.get('DEEPINFER_910C_GREEDY_SHADOW')=='1':
        from .dspark_greedy_shadow910c import GreedyShadow
        shadow=GreedyShadow()
        model.greedy_shadow_record=shadow.stats
        if os.environ.get('DEEPINFER_910C_PERSISTENT_STATE_AUDIT')=='1':
            from .fixed24_persistent_audit910c import PersistentStateAudit
            persistent_state_audit=PersistentStateAudit(limit=shadow.stats['limit'])
            shadow.stats['persistent_state']=persistent_state_audit.stats
    variable=os.environ.get('DEEPINFER_ENABLE_MTP_VARLEN')=='1'
    policy=None
    if variable:
        from .dspark_graphs910c import DSparkTargetGraphs
        from .dspark_confidence910c import DSparkConfidenceScheduler
        if model.dspark_graphs is None:
            model.dspark_graphs=DSparkTargetGraphs(model)
        stats.update(verification_mode='confidence-scheduled',query_length_histogram={},
                     selected_total_queries=0,padded_target_queries=0)

    def output(slot,tokens):
        import time
        index,request=owned[slot]
        if finished[slot]:return
        for token in tokens:
            generated[slot].append(token)
            stats['emitted_target_tokens']+=1
            emit({'type':'token','index':index,'token_id':token,'step':len(generated[slot])-1,
                  'elapsed':time.monotonic()-started})
            if (len(generated[slot])==maximum[slot] or
                    (not request.get('ignore_eos',False) and token==eos)):
                finished[slot]=True
                break
        if finished[slot]:
            emit({'type':'done','index':index,'prompt_tokens':len(request['token_ids']),
                'cached_tokens':hits[slot],'output_tokens':len(generated[slot]),
                'token_ids':generated[slot],'finish_reason':
                    finish_reason(generated[slot][-1], len(generated[slot]), maximum[slot],
                        ignore_eos=request.get('ignore_eos', False), eos_token_id=eos)})

    for slot,token in enumerate(target_tokens(runner,logits)):
        output(slot,[token])
    first_proposal=True
    fast_enabled=(os.environ.get('DEEPINFER_910C_ZOS_FAST')=='1' and shadow is None and variable)
    async_state=None
    async_error=False
    device_target_enabled=(fast_enabled and os.environ.get('DEEPINFER_910C_ZOS_TARGET_DEVICE')=='1')
    device_c2_enabled=(fast_enabled and os.environ.get('DEEPINFER_910C_ZOS_C2_DEVICE')=='1')
    device_c2_check=(device_c2_enabled and os.environ.get('DEEPINFER_910C_ZOS_C2_CHECK')=='1')
    device_window_enabled=(fast_enabled and os.environ.get('DEEPINFER_910C_ZOS_ENGRAM_WINDOW')=='1')
    device_window_check=(device_window_enabled and os.environ.get('DEEPINFER_910C_ZOS_ENGRAM_WINDOW_CHECK')=='1')
    stats['zos_fast']={'enabled':fast_enabled,'prelaunches':0,'consumed':0,'verified':0,'fallback_steps':0,'errors':0,
                       'device_target_enabled':device_target_enabled,'device_target_calls':0,
                       'device_c2_enabled':device_c2_enabled,'device_c2_calls':0,'device_c2_errors':0,
                       'device_c2_check':device_c2_check,'device_c2_checked_rows':0,
                       'device_window_enabled':device_window_enabled,'device_window_calls':0,
                       'device_window_checks':0,'device_window_errors':0}
    prelaunched=None
    prelaunch_audit=(os.environ.get('DEEPINFER_910C_SPEC_DRAFT_AUDIT')=='1' and shadow is not None)
    if prelaunch_audit and persistent_state_audit is None:
        raise RuntimeError('draft prelaunch audit requires persistent device state audit')
    while True:
        bases=[model.next_positions[s] for s in range(count)]
        golden=[row[-1] for row in generated]
        scheduled=policy.schedule() if fast_enabled and policy is not None else None
        if fast_enabled:
            local_eligible=(scheduled==[width]*count and width==6 and
                all(req.get('ignore_eos',False) for _,req in owned) and
                all(base+6+draft.nextn<=model.max_tokens for base in bases))
            busy,eligible,failed=runner._global_zos_vote(not all(finished),local_eligible,async_error)
            if failed:raise RuntimeError('a rank rejected asynchronous draft state')
            if not busy:break
        else:
            if not runner._global_busy(not all(finished)):break
            eligible=False
        draft_start=time.perf_counter()
        device_draft_inputs=None
        if os.environ.get('DEEPINFER_910C_PERSISTENT_DRAFT_INPUT_AUDIT')=='1' and shadow is not None:
            if persistent_state_audit is None:
                raise RuntimeError('device draft input audit requires persistent state audit')
            device_draft_inputs=persistent_state_audit.draft_inputs(golden,bases,allow_after_limit=prelaunched is not None)
        if prelaunched is not None:
            if fast_enabled:
                stats['zos_fast']['consumed']+=1
            elif device_draft_inputs is None:
                raise RuntimeError('prelaunched draft lacks checked device state')
            else:
                persistent_state_audit.stats['prelaunch_consumed']=persistent_state_audit.stats.get('prelaunch_consumed',0)+1
            proposals,confidence=prelaunched
            prelaunched=None
        elif device_draft_inputs is None:
            proposals,confidence=draft.propose(golden,bases,list(range(count)))
        else:
            proposals,confidence=draft.propose(*device_draft_inputs,list(range(count)))
            persistent_state_audit.stats['draft_device_graph_calls']=persistent_state_audit.stats.get('draft_device_graph_calls',0)+1
        proposal_values=proposals.cpu().tolist()
        stats['confidence_head_executed']=stats['markov_head_executed']=True
        draft_seconds=time.perf_counter()-draft_start
        stats['draft_and_sampling_wall_seconds']+=draft_seconds
        confidence_values=confidence.cpu().tolist() if variable or first_proposal else None
        if first_proposal:
            stats['first_draft_token_ids']=proposal_values
            stats['first_confidence']=confidence_values
            first_proposal=False
        if variable:
            if policy is None:
                curve=model.dspark_graphs.profile(golden,proposal_values)
                policy=DSparkConfidenceScheduler(curve)
                stats['verification_budget_policy']='diagnostic-fixed5' if policy.ablation_fixed5 else 'original-measured-cost'
                stats['measured_target_curve_ms']=curve
                stats['variable_target_prefix_validation']=model.dspark_graphs.prefix_validation
            lengths=scheduled if scheduled is not None else policy.schedule()
        else:lengths=[width]*count
        chunks=[[g,*row[:length-1]] for g,row,length in zip(golden,proposal_values,lengths)]
        shadow_call=bool(shadow is not None and shadow.wants(lengths,owned,variable))
        persistent_call=False
        if shadow_call:
            shadow_remaining=[maximum[i]-len(generated[i]) for i in range(count)]
            shadow_finished=list(finished)
            if persistent_state_audit is not None:
                persistent_call=persistent_state_audit.before(model,model.dspark_graphs.full,
                    proposals,golden,shadow_remaining,shadow_finished,chunks)
        elif persistent_state_audit is not None:
            persistent_state_audit.interrupt()
        if fast_enabled and not eligible:
            async_state=None
            stats['zos_fast']['fallback_steps']+=1
        if eligible and async_state is None:
            from .async_draft_state910c import AsyncDraftState
            init_error=None
            try:
                async_state=AsyncDraftState(model,golden,
                    [maximum[i]-len(generated[i]) for i in range(count)],finished)
            except Exception as exc:
                init_error=repr(exc)
            # Initialization can fail rank-locally; vote before target/draft collectives.
            ready=torch.tensor([int(init_error is None)],dtype=torch.int32,device=model.device)
            dist.all_reduce(ready,op=dist.ReduceOp.MIN)
            if not bool(ready.item()):raise RuntimeError('async state initialization failed on at least one rank: '+str(init_error))
        target_start=time.perf_counter()
        if variable:
            frame=model.dspark_graphs.full
            use_device_target=device_target_enabled and eligible and lengths==[6]*count
            use_device_window=device_window_enabled and eligible and lengths==[6]*count
            if use_device_window:
                if frame.device_engram_fast_override is not None:raise RuntimeError('unconsumed device window')
                before_window=frame.device_engram_fast_calls
                before_check=frame.device_engram_fast_checked
                frame.device_engram_fast_override=(async_state.prepare_engram_window(proposals),device_window_check)
            if use_device_target:
                if frame.device_input_override is not None:raise RuntimeError('unconsumed device target input')
                before=frame.device_input_override_applied
                frame.device_input_override=async_state.target_graph_input(proposals)
            try:
                target_logits=model.dspark_graphs.run(chunks,[[g,*row] for g,row in zip(golden,proposal_values)])
                if use_device_target:
                    if model.dspark_graphs.last_frame is not frame or frame.device_input_override_applied!=before+1:
                        raise RuntimeError('device target input was not consumed by the full graph')
                    stats['zos_fast']['device_target_calls']+=1
                if use_device_window:
                    if (model.dspark_graphs.last_frame is not frame or
                            frame.device_engram_fast_calls!=before_window+1 or
                            frame.device_engram_fast_checked!=before_check+int(device_window_check)):
                        raise RuntimeError('device Engram window was not consumed')
                    stats['zos_fast']['device_window_calls']+=1
                    stats['zos_fast']['device_window_checks']+=int(device_window_check)
            finally:
                if use_device_target:frame.device_input_override=None
                if use_device_window:frame.device_engram_fast_override=None
            if eligible:
                pending=async_state.target_tokens.queue(target_tokens(runner,target_logits,defer_host=True))
                draft_golden,draft_positions=async_state.queue(proposals,pending.device_result)
                prelaunched=draft.propose(draft_golden,draft_positions,list(range(count)))
                stats['zos_fast']['prelaunches']+=1
                if device_c2_enabled:
                    async_state.queue_c2(frame)
                values=pending.wait()
                if use_device_window and not async_state.verify_engram_window(bases):
                    stats['zos_fast']['device_window_errors']+=1
                    raise RuntimeError('device Engram window status or position invalid')
            else:
                values=target_tokens(runner,target_logits,return_device=shadow_call)
                if shadow_call:values,shadow_target_device=values
            predictions=model.dspark_graphs.split_predictions(values,lengths)
            frame=model.dspark_graphs.last_frame
            stats['selected_total_queries']+=sum(lengths)
            stats['padded_target_queries']+=frame.tokens-sum(lengths)
        else:
            predictions=target_tokens(runner,model.verify_chunks(chunks),return_device=shadow_call)
            if shadow_call:predictions,shadow_target_device=predictions
            frame=model._verify_graph
        target_seconds=time.perf_counter()-target_start
        stats['target_and_sampling_wall_seconds']+=target_seconds
        prelaunch_voted=False
        proposal_for_shadow=proposals
        if prelaunch_audit:
            # All EP ranks vote before launching TP draft collectives. Never
            # use a locally eligible decision when the other DP is ineligible.
            # Current full verification can commit at most six tokens; the
            # following five-token draft must stay within the native context.
            capacity=all(base+6+draft.nextn<=model.max_tokens for base in bases)
            audit_eligible=torch.tensor([int(persistent_call and capacity)],dtype=torch.int32,device=model.device)
            dist.all_reduce(audit_eligible,op=dist.ReduceOp.MIN)
            if bool(audit_eligible.item()):
                state=persistent_state_audit.prelaunch_state(model,frame,shadow_target_device)
                ready=torch.tensor([int(state is not None)],dtype=torch.int32,device=model.device)
                dist.all_reduce(ready,op=dist.ReduceOp.MIN)
                if not bool(ready.item()):
                    raise RuntimeError('a rank failed to prepare the diagnostic draft prelaunch')
                # The draft graph reuses its output buffer. Preserve the old
                # proposal on device for the current iteration's shadow check.
                proposal_for_shadow=persistent_state_audit.staged_draft
                prelaunched=draft.propose(state.golden,state.pos,list(range(count)))
                persistent_state_audit.stats['prelaunch_graph_calls']=persistent_state_audit.stats.get('prelaunch_graph_calls',0)+1
                prelaunch_voted=True
        commit_start=time.perf_counter()
        stats['proposals']+=1
        commits=[]
        if shadow_call or eligible:shadow_accepted=[];shadow_emitted=[]
        for slot,(candidates,targets,length) in enumerate(zip(proposal_values,predictions,lengths)):
            if finished[slot]:
                if shadow_call or eligible:shadow_accepted.append(0);shadow_emitted.append([])
                commits.append(0);continue
            accepted=0
            for candidate,target in zip(candidates[:length-1],targets):
                if candidate!=target:break
                accepted+=1
            stats['draft_tokens_verified']+=length-1
            stats['draft_tokens_accepted']+=accepted
            if variable:
                histogram=stats['query_length_histogram']
                histogram[str(length)]=histogram.get(str(length),0)+1
            commit=min(accepted+1,maximum[slot]-len(generated[slot]))
            before=len(generated[slot])
            output(slot,targets[:commit])
            commits.append(len(generated[slot])-before)
            if shadow_call or eligible:
                shadow_accepted.append(accepted)
                shadow_emitted.append(list(generated[slot][before:]))
        device_pending=None
        if eligible and device_c2_enabled:
            device_pending,c2_ok,checked_rows=async_state.settle_c2(bases,commits,frame,check=device_c2_check)
            stats['zos_fast']['device_c2_checked_rows']+=checked_rows
            async_ok=async_state.settle(shadow_accepted,commits,shadow_emitted)
            stats['zos_fast']['device_c2_calls']+=1
            if not c2_ok:
                stats['zos_fast']['device_c2_errors']+=1
            async_error=not (async_ok and c2_ok)
            if async_error:device_pending=None
            frame.commit_prefixes(bases,commits,device_pending=device_pending)
        else:
            frame.commit_prefixes(bases,commits)
            if eligible:
                async_error=not async_state.settle(shadow_accepted,commits,shadow_emitted)
        if eligible:
            if async_error:stats['zos_fast']['errors']+=1
            else:stats['zos_fast']['verified']+=1
        commit_seconds=time.perf_counter()-commit_start
        stats['commit_and_emit_wall_seconds']+=commit_seconds
        if variable:
            events=frame.last_forward_events
            forward_ms=events[0].elapsed_time(events[1])
            policy.observe_forward_ms(model.dspark_graphs.last_graph_size,forward_ms)
            policy.observe_sampler_ms(max(0.,1000*(draft_seconds+target_seconds+commit_seconds)-forward_ms))
            policy.observe_confidence(confidence_values)
            stats['confidence_fp32_roundoff_clamps']=policy.confidence_roundoff_clamps
        if shadow_call:
            shadow.compare(proposal_for_shadow,shadow_target_device,proposal_values,predictions,
                           shadow_remaining,shadow_finished,shadow_accepted,commits,shadow_emitted)
            if persistent_call:
                persistent_state_audit.after(model,frame,shadow_target_device,commits,
                    shadow_accepted,shadow_emitted,generated,maximum,finished)
        if prelaunch_voted:
            valid=torch.tensor([int(persistent_state_audit.stats['failures']==0 and shadow.stats['failures']==0)],dtype=torch.int32,device=model.device)
            dist.all_reduce(valid,op=dist.ReduceOp.MIN)
            if not bool(valid.item()):
                raise RuntimeError('draft prelaunch audit failed on at least one rank')
        if runner.tp_rank==0 and stats['proposals']%32==0:
            runner.engine.progress({'stage':'dspark','dp_rank':runner.dp_rank,
                'generated_tokens':[len(row) for row in generated],**stats})
    return generated
