# SPDX-License-Identifier: Apache-2.0
"""Native DSpark greedy rejection sampling for the fixed DP2/TP8 service.

Candidates come exclusively from checkpoint mtp layers, never from text
lookup. All accepted outputs are checked against the full target model.
"""
import torch
import torch.distributed as dist
import time
import os


def target_tokens(runner,logits):
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
    return result.cpu().tolist()


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
                    'length' if len(generated[slot])==maximum[slot] else 'stop'})

    for slot,token in enumerate(target_tokens(runner,logits)):
        output(slot,[token])
    first_proposal=True
    while runner._global_busy(not all(finished)):
        bases=[model.next_positions[s] for s in range(count)]
        golden=[row[-1] for row in generated]
        draft_start=time.perf_counter()
        proposals,confidence=draft.propose(golden,bases,list(range(count)))
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
                stats['measured_target_curve_ms']=curve
                stats['variable_target_prefix_validation']=model.dspark_graphs.prefix_validation
            lengths=policy.schedule()
        else:lengths=[width]*count
        chunks=[[g,*row[:length-1]] for g,row,length in zip(golden,proposal_values,lengths)]
        target_start=time.perf_counter()
        if variable:
            values=target_tokens(runner,model.dspark_graphs.run(chunks,[[g,*row] for g,row in zip(golden,proposal_values)]))
            predictions=model.dspark_graphs.split_predictions(values,lengths)
            frame=model.dspark_graphs.last_frame
            stats['selected_total_queries']+=sum(lengths)
            stats['padded_target_queries']+=frame.tokens-sum(lengths)
        else:
            predictions=target_tokens(runner,model.verify_chunks(chunks))
            frame=model._verify_graph
        target_seconds=time.perf_counter()-target_start
        stats['target_and_sampling_wall_seconds']+=target_seconds
        commit_start=time.perf_counter()
        stats['proposals']+=1
        commits=[]
        for slot,(candidates,targets,length) in enumerate(zip(proposal_values,predictions,lengths)):
            if finished[slot]:
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
        frame.commit_prefixes(bases,commits)
        commit_seconds=time.perf_counter()-commit_start
        stats['commit_and_emit_wall_seconds']+=commit_seconds
        if variable:
            events=frame.last_forward_events
            forward_ms=events[0].elapsed_time(events[1])
            policy.observe_forward_ms(model.dspark_graphs.last_graph_size,forward_ms)
            policy.observe_sampler_ms(max(0.,1000*(draft_seconds+target_seconds+commit_seconds)-forward_ms))
            policy.observe_confidence(confidence_values)
            stats['confidence_fp32_roundoff_clamps']=policy.confidence_roundoff_clamps
        if runner.tp_rank==0 and stats['proposals']%32==0:
            runner.engine.progress({'stage':'dspark','dp_rank':runner.dp_rank,
                'generated_tokens':[len(row) for row in generated],**stats})
    return generated
