"""将冻结证据解析为面向读者的单文件报告。只呈现，不运行性能实验。"""
from pathlib import Path
from html import escape as esc
import hashlib
import json

U = Path(__file__).resolve().parent
ROOT = U.parent
DEST = ROOT / 'request-flow/report.html'


def e(value):
    return esc(str(value), quote=True)


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join('<th>'+e(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>'+str(x)+'</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def paragraph(label, text):
    return '<p><b>'+e(label)+'：</b>'+e(text)+'</p>'


def build():
    F = json.loads((ROOT/'request-flow/data.json').read_text())
    D = json.loads((ROOT/'map/combined-map.json').read_text())
    scenes = [s for s in D['baseline']['bench_scenes'] if s['phase']=='formal']
    scene_index = {(s['framework'],s['n'],s['c']):s for s in scenes}
    assert len(scenes)==56
    ttft_sg = sum(s['sglang']['ttft_s'] < s['vllm']['ttft_s'] for s in F['summaries'])
    tpot_v = sum(s['vllm']['endpoint_tpot_ms'] < s['sglang']['endpoint_tpot_ms'] for s in F['summaries'])
    tps_v = sum(scene_index['vllm',s['n'],s['c']]['metrics']['output_tps']['median'] > scene_index['sglang',s['n'],s['c']]['metrics']['output_tps']['median'] for s in F['summaries'])
    assert (ttft_sg,tpot_v,tps_v)==(18,28,28)
    flows = []
    # Each stage gets short, comparable interpretations rather than serialized cards.
    takeaways = {
        'entry':'同样输入token IDs，避免把分词差异混入计分。收到HTTP请求、后端接纳和设备开始执行仍是三个时间点。',
        'queue':'TTFT长，可能是请求还在等位置，也可能是prompt计算慢；只有入队/接纳/首次执行的同请求事件才能区分。',
        'schedule':'关注“一轮让多少prompt进入、已经生成的请求要等多久”。两个框架都填8192，不保证得到相同的批次。',
        'state':'GDN保留的是递归状态，不是把所有层都当普通KV。换池布局或复用前缀时，必须同时维持这两种历史。',
        'metadata':'图回放能减少发射工作，但图尺寸大于有效工作时可能增加补齐成本。字段没有记录，不能按0算。',
        'prefill':'prefill不仅处理输入，还建立后续解码使用的状态。长prompt可能分多轮完成；mixed会让已有decode穿插其中。',
        'layers':'要按同一逻辑范围比较：一个融合算子可能替代多个小算子。不能只比算子名称或调用次数。',
        'sample':'采样完成不代表用户已经看到字。token需要提交、转文本和发送；有些token还不会立刻产生非空文本。',
        'decode':'下一轮计算依赖上一轮的token和已提交状态。CPU准备可与设备交错，真实重叠需设备事件证明。',
        'ipc':'worker发消息、前端收到消息和处理完成必须用同一消息ID串起来；只凭时间接近不能算纯IPC耗时。',
        'text':'反分词通常嵌套在输出处理里。把反分词时间再加到整个输出处理时间上，会重复计时。',
        'sse':'生成器产出字符串、调用ASGI、await返回、客户端收到是不同阶段。旧hook只记录创建coroutine，不能解释成网络发送耗时。',
        'eof':'客户端结束只是用户侧边界；服务端还可能有不再产生token的尾步。尾步不能全部归到用户等待。',
    }
    for i,c in enumerate(F['components'],1):
        mechanisms = c['mechanisms']
        difference = next((m['framework_difference'] for m in mechanisms if m.get('framework_difference')), c['principle'])
        source_cards=[]
        for fw in ['vllm','sglang']:
            refs=[a for a in c['sources'] if (a['framework'] or '').startswith('vllm')] if fw=='vllm' else [a for a in c['sources'] if a['framework']=='sglang']
            items=[]
            for a in refs[:3]:
                if not a['url']:continue
                names=[v.get('qualified_name','') for v in a['symbols'] if isinstance(v,dict)]
                items.append('<li><a href="'+e(a['url'])+'">'+e(a['path'])+'</a><span class="source-meta">行 '+e(a['lines'])+' · commit '+e(str(a['commit'])[:12])+'</span><span class="source-meta">'+e('、'.join(names[:3]))+'</span></li>')
            source_cards.append('<div><b>'+('vLLM / Ascend' if fw=='vllm' else 'SGLang')+'</b><ul class="sources">'+''.join(items)+'</ul></div>')
        flows.append('<article class="stage" id="stage-'+e(c['id'])+'"><div class="stage-heading"><span class="number">'+str(i).zfill(2)+'</span><h3>'+e(c['name'])+'</h3></div><p class="lead-small">'+e(c['principle'])+'</p><div class="compare"><div class="v"><b>vLLM Ascend</b><p>'+e(c['vllm'])+'</p></div><div class="s"><b>SGLang Ascend</b><p>'+e(c['sglang'])+'</p></div></div>'+paragraph('机制差异',difference)+'<div class="interpret">'+paragraph('怎样读这一阶段',takeaways[c['id']])+paragraph('实测到了哪里',c['timing'])+'</div><details><summary>查看本阶段的精确源码位置</summary><div class="compare">'+''.join(source_cards)+'</div></details></article>')
    summary_rows=[]
    for s in F['summaries']:
        cells=[f"{s['n']:,} / C{s['c']}"]
        for metric,unit in [('ttft_s','s'),('endpoint_tpot_ms','ms')]:
            v,sg=s['vllm'][metric],s['sglang'][metric]
            cells.append(f'<b class="'+('vtext' if v<sg else 'stext')+f'">{v:.3f} / {sg:.3f}</b> '+unit)
        v,sg=(scene_index[fw,s['n'],s['c']]['metrics']['output_tps']['median'] for fw in ['vllm','sglang'])
        cells.append(f'<b class="vtext">{v:.2f} / {sg:.2f}</b> tok/s')
        spreads=[scene_index[fw,s['n'],s['c']]['metrics']['output_tps']['relative_range']*100 for fw in ['vllm','sglang']]
        cells.append(f'{spreads[0]:.2f}% / {spreads[1]:.2f}%')
        summary_rows.append(cells)
    decode_rows=[]
    for x in F['decode']:
        z=x['intervals_ms']
        decode_rows.append([e(x['framework']+' / rank '+str(x['rank'])),f"{x['fence_ms']:.3f}",f"{z['compute_union_us']:.3f}",f"{z['collective_envelope_union_us']:.3f}",f"{z['collective_compute_overlap_us']:.3f}",f"{x['uncovered_by_compute_collective_ms']:.3f}",f"{z['transfer_named_union_us']:.3f}",f"{z['notify_wait_union_us']:.3f}"])
    phase_rows=[]
    for x in F['phases']:
        z=x['intervals_us'];phase_rows.append([e(x['framework']),('预填充' if x['phase']=='prefill' else '输入与解码混合'),e(x['step_id']),str(x['collective_count']),*[f'{float(z[k])/1000:.3f}' for k in ['compute_subset_union','collective_envelope_union','transfer_named_union','notify_wait_union']]])
    chain_rows=[
        ['最后输出进入前端处理','vLLM','最后一份引擎输出的时间标记 → 输出处理函数进入','0.752','是标记到前端进入的包络，不能单独称作IPC传输。'],
        ['输出处理整体','vLLM','输出处理函数进入 → 返回','1.841','包含下方反分词调用；两行不能相加。'],
        ['最后一次反分词','vLLM','反分词进入 → 返回','0.542','CPU插桩调用时间，包含观察器影响。'],
        ['结束标记产出','vLLM','输出处理返回 → API产出SSE结束标记','1.454','产出SSE不等于ASGI已发送。'],
        ['客户端读完响应','vLLM','收到DONE → HTTP响应体结束','0.270','两端点都在客户端时间域。'],
        ['最后一次输出提交调用','SGLang','开始向输出IPC发送 → 调用返回','0.139','同一次调用，不能解释成对端已收到。'],
        ['最后输出到反分词结果','SGLang','IPC发送返回 → 同请求最后反分词批输出','0.906','只闭合同请求包络，尚缺消息ID。'],
        ['客户端读完响应','SGLang','收到DONE → HTTP响应体结束','0.110','两端点都在客户端时间域。'],
        ['结束后仍有尾部提交','SGLang','客户端响应体结束 → 零token尾部提交','12.775','采用已有时钟对应收据；尾部没有增加用户等待，不计作可省的E2E。'],
    ]
    candidates=[]
    for x in D['design']['opportunities']:
        candidates.append('<article class="candidate"><span class="tag">'+e(x['id'])+' · '+e(x['project'])+'</span><h3>'+e(x['name'])+'</h3>'+paragraph('想改什么',x['mechanism'])+paragraph('已有观察',x['local_observation'])+paragraph('收益怎样判断',x['benefit_bound'])+paragraph('还要验证',x['validation'])+'<p class="small">'+e(x['limits'])+'</p></article>')
    transfer=[]
    for x in D['mechanisms']['transfer_from_strata_deepinfer']['principles']:
        links=' · '.join('<a href="'+e(a['url'])+'">'+e(a['path'])+'</a>' for a in x.get('source_anchors',[]) if a.get('url'))
        transfer.append('<article class="candidate"><span class="tag">'+e(x['project'])+' · '+e(x['evidence_class'])+'</span><h3>'+e(x['name'])+'</h3>'+paragraph('原理',x['mechanism'])+paragraph('对当前模型是否适用',x['current_model_fit'])+paragraph('移植前的门槛',x['validation_before_transfer'])+'<p class="small">'+e(x['limits'])+'</p><details><summary>公开来源</summary><p class="sources">'+links+'</p></details></article>')
    # Only structured observations needed by the interactive readers, no source-card JSON dump.
    client=dict(requests=F['requests'],summaries=F['summaries'],steps=F['diagnostic_steps'],
                profiles=[{k:x[k] for k in ['framework','n','c','rank','phase','window_label','operator_categories','communication_by_family']} for x in F['profiles']])
    replacements={
        '__STAGES__':''.join(flows),
        '__CONFIG_TABLE__':table(['比较项目','vLLM部署','SGLang部署','怎样影响比较'],[[e(x['parameter']),e(x['vllm']),e(x['sglang']),e(x['interpretation'])] for x in D['baseline']['configuration_rows']]),
        '__PERFORMANCE_TABLE__':table(['输入token / 并发','首文本等待，越小越好<br>V / S','端点每token耗时，越小越好<br>V / S','整波输出吞吐，越大越好<br>V / S','吞吐三轮极差 / 中位数<br>V / S'],summary_rows),
        '__DECODE_TABLE__':table(['框架 / rank','窗口','计算并集','通信包络并集','计算通信重叠','二者未覆盖','传输命名任务','等待通知任务'],decode_rows),
        '__PHASE_TABLE__':table(['框架','阶段','原始步标识','collective数','计算子集','通信包络','传输命名任务','等待通知任务'],phase_rows),
        '__CHAIN_TABLE__':table(['人能读懂的阶段','框架','真实测量边界','耗时ms','怎么理解'],[[e(c) for c in row] for row in chain_rows]),
        '__CANDIDATES__':''.join(candidates[:5]),
        '__TRANSFER__':''.join(transfer),
        '__CLIENT_DATA__':json.dumps(client,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'),
    }
    # Headers can contain explicit line breaks, never source-derived markup.
    for key in ['__PERFORMANCE_TABLE__']:
        replacements[key]=replacements[key].replace('&lt;br&gt;','<br>')
    html=(U/'human_report_template.html').read_text()
    for k,v in replacements.items():html=html.replace(k,v)
    assert '__CLIENT_DATA__' not in html
    DEST.write_text(html)
    (ROOT/'request-flow/index.html').write_text(html)
    print(json.dumps({'file':str(DEST),'bytes':DEST.stat().st_size,'requests':len(F['requests']),'stages':13,'formal_scene_pairs':28,'data_sha256':hashlib.sha256((ROOT/'request-flow/data.json').read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=='__main__':build()
