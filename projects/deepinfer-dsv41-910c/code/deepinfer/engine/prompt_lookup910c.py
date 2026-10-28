# SPDX-License-Identifier: Apache-2.0
"""Input-only n-gram drafting with exact native target verification.

This module reads only request input IDs and accepted generated IDs. Every
emitted token comes from the full target model's greedy logits. Rejected
lookahead is repaired with the same packed target and native cache manager.
"""
import time

import torch
import torch.distributed as dist


class PromptLookup:
    def __init__(self,tokens,draft_tokens):
        self.tokens=tuple(tokens)
        self.draft_tokens=draft_tokens
        self.maps={}
        for order in (8,4):
            table={}
            for start in range(len(tokens)-order-draft_tokens+1):
                table.setdefault(tuple(tokens[start:start+order]),start+order)
            self.maps[order]=table

    def propose(self,tail):
        for order,table in self.maps.items():
            if len(tail)>=order:
                start=table.get(tuple(tail[-order:]))
                if start is not None:
                    return list(self.tokens[start:start+self.draft_tokens])
        return None


def sample(runner,logits):
    shape=logits.shape[:-1]
    values=logits.argmax(-1).long() if runner.tp_rank==0 else torch.empty(shape,dtype=torch.int64,device=runner.model.device)
    dist.broadcast(values,runner.root,group=runner.tp_group)
    return values.cpu().tolist()


@torch.inference_mode()
def generate(runner,owned,logits,hits,emit,started,draft_tokens=63):
    model=runner.model
    width=draft_tokens+1
    if len(owned)!=runner.slots or runner.slots!=4:
        raise ValueError('native lookup profile requires four requests per DP')
    model.use_decode_graph=False
    model.mc2_global_bs=((runner.slots*width+runner.tp_size-1)//runner.tp_size)*dist.get_world_size()
    maximum=[request['max_tokens'] for _,request in owned]
    generated=[[] for _ in owned]
    finished=[False]*len(owned)
    noise=model.cfg.get('dspark_noise_token_id',0)
    if not 0<=noise<model.embedding.shape[0] or noise in {model.reader.config.get('image_token_id'),model.reader.config.get('image_pad_token_id')}:
        noise=0
    stats={'method':'prompt_lookup','max_draft_tokens':draft_tokens,
        'verification_query_tokens':width,'verification_forwards':0,'repair_forwards':0,
        'lookup_hits':0,'lookup_misses':0,'draft_tokens_verified':0,'draft_tokens_accepted':0,
        'emitted_target_tokens':0}
    model.prompt_lookup_stats=stats

    def output(slot,tokens):
        index,request=owned[slot]
        for token in tokens:
            if len(generated[slot])>=maximum[slot]:
                break
            generated[slot].append(token)
            stats['emitted_target_tokens']+=1
            emit({'type':'token','index':index,'token_id':token,'step':len(generated[slot])-1,
                  'elapsed':time.monotonic()-started})
        if len(generated[slot])>=maximum[slot] and not finished[slot]:
            finished[slot]=True
            emit({'type':'done','index':index,'prompt_tokens':len(request['token_ids']),
                'cached_tokens':hits[slot],'output_tokens':len(generated[slot]),
                'token_ids':generated[slot],'finish_reason':'length'})

    first=sample(runner,logits)
    for slot,token in enumerate(first):
        output(slot,[token])
    indices=[PromptLookup(request['token_ids'],draft_tokens) for _,request in owned]

    while runner._global_busy(not all(finished)):
        bases=[model.next_positions[slot] for slot in range(runner.slots)]
        chunks=[]
        was_active=[not value for value in finished]
        for slot,(_,request) in enumerate(owned):
            if finished[slot]:
                chunks.append([noise]*width)
                continue
            tail=(request['token_ids'][-8:]+generated[slot])[-8:]
            proposal=indices[slot].propose(tail)
            if proposal is None:
                proposal=[noise]*draft_tokens;stats['lookup_misses']+=1
            else:
                stats['lookup_hits']+=1
            chunks.append([generated[slot][-1],*proposal])
        targets=sample(runner,model.verify_chunks(chunks))
        stats['verification_forwards']+=1
        commits=[]
        for slot,(chunk,predictions) in enumerate(zip(chunks,targets)):
            if not was_active[slot]:
                commits.append(0)
                continue
            accepted=0
            for candidate,predicted in zip(chunk[1:],predictions):
                if candidate!=predicted:
                    break
                accepted+=1
            stats['draft_tokens_verified']+=draft_tokens
            stats['draft_tokens_accepted']+=accepted
            commit=min(accepted+1,maximum[slot]-len(generated[slot]))
            commits.append(commit)
            output(slot,predictions[:commit])

        active_global=runner._global_busy(not all(finished))
        if not active_global:
            # No subsequent consumer needs lookahead state. Keep host counters
            # and history at the committed boundary for the final audit.
            for slot,(base,commit) in enumerate(zip(bases,commits)):
                model.next_positions[slot]=base+commit
                if model.histories:
                    del model.histories[slot].history[base+commit:]
            break

        if runner._global_busy(any(commit!=width for commit in commits)):
            # Future SWA writes lie outside the live128-token past because the
            # ring covers >128+width. Restored lengths mask future compressed
            # rows. Recompute only committed prefixes to restore exact C2
            # pending values and CPU Engram history after rejection.
            frame=model._verify_graph
            for slot,base in enumerate(bases):
                model.next_positions[slot]=base
                if model.histories:
                    del model.histories[slot].history[base:]
            for source,pair in frame.even_before.items():
                pending=model.pending.setdefault(source,{})
                for slot,base in enumerate(bases):
                    if base%2:
                        pending[slot]=tuple(t[slot:slot+1].clone() for t in pair)
                    else:
                        pending.pop(slot,None)
            slots=[slot for slot,commit in enumerate(commits) if commit]
            accepted_chunks=[chunks[slot][:commits[slot]] for slot in slots]
            if not slots:
                runner._reset_scratch()
                slots,accepted_chunks=[runner.slots],[[noise]]
            model.forward_tokens(accepted_chunks,slots)
            frame.prepared=False
            stats['repair_forwards']+=1
        if runner.tp_rank==0 and stats['verification_forwards']%4==0:
            runner.engine.progress({'stage':'prompt_lookup','dp_rank':runner.dp_rank,
                'generated_tokens':[len(row) for row in generated],**stats})
    return generated
