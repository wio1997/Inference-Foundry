"""离线复算脱敏请求、官方汇总及对比表。只用Python标准库，不运行推理。"""
from pathlib import Path
import csv, hashlib, json, statistics

ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def quantile(vals,q):
    vals=sorted(vals);i=(len(vals)-1)*q;a=int(i);b=min(a+1,len(vals)-1)
    return vals[a]+(vals[b]-vals[a])*(i-a)
def main():
    phases=load(ROOT/'rounds.json')['rounds'];count=0;identities={}
    for item in load(ROOT/'evidence/manifest.json'):
        assert hashlib.sha256((ROOT/item['artifact']).read_bytes()).hexdigest()==item['archived_sha256'],item['artifact']
    formal={}
    for phase in phases:
        p=ROOT/phase['evidence'];rows=load(p/'requests.json');official=load(p/'official.json')
        count+=len(rows);assert len(rows)==(32 if phase['phase']=='formal' else 16)
        assert len({str(r['data_id']) for r in rows})==len(rows)
        for r in rows:
            assert r['success'] and r['input_tokens']==phase['input_tokens'] and r['output_tokens']==1024
            t=r['time_points_s'];assert len(t)>=2 and all(b>=a for a,b in zip(t,t[1:]))
            key=(phase['phase'],phase['input_tokens'],str(r['data_id']))
            if key in identities:assert identities[key]==r['input_sha256'],key
            identities[key]=r['input_sha256']
        duration=max(r['time_points_s'][-1] for r in rows)-min(r['time_points_s'][0] for r in rows)
        assert abs(duration-phase['duration_s'])<1e-6
        get=lambda key:float(str(official[key]['total']).split()[0])
        assert get('Success Requests')==len(rows) and get('Failed Requests')==0
        assert get('Total Generated Tokens')==len(rows)*1024
        assert get('Total Input Tokens')==len(rows)*phase['input_tokens']
        assert abs(get('Benchmark Duration')/1000-duration)<1e-6
        assert abs(len(rows)*1024/duration-get('Output Token Throughput'))<.000051
        assert abs(len(rows)*phase['input_tokens']/duration-get('Input Token Throughput'))<.000051
        assert abs(len(rows)*(phase['input_tokens']+1024)/duration-get('Total Token Throughput'))<.000051
        events=sorted([(r['time_points_s'][0],1) for r in rows]+[(r['time_points_s'][-1],-1) for r in rows])
        active=peak=0
        for _,delta in events:active+=delta;peak=max(peak,active)
        assert peak==get('Max Concurrency')==phase['observed_peak_concurrency']==16
        average=sum(r['time_points_s'][-1]-r['time_points_s'][0] for r in rows)/duration
        assert abs(average-get('Concurrency'))<.000051
        stats={r['Performance Parameters']:r for r in csv.DictReader((p/'official.csv').open())}
        for name,values in [
            ('TTFT',[(r['time_points_s'][1]-r['time_points_s'][0])*1000 for r in rows]),
            ('E2EL',[(r['time_points_s'][-1]-r['time_points_s'][0])*1000 for r in rows]),
            ('TPOT',[(r['time_points_s'][-1]-r['time_points_s'][1])/1023*1000 for r in rows])]:
            for k,v in [('Average',statistics.mean(values)),('Min',min(values)),('Max',max(values)),
                        ('Median',statistics.median(values)),('P75',quantile(values,.75)),('P90',quantile(values,.9)),('P99',quantile(values,.99))]:
                assert abs(v-float(stats[name][k].split()[0]))<.051,(phase['round_id'],name,k,v,stats[name][k])
        if phase['phase']=='formal':
            m=phase['metrics'];assert abs(m['aggregate_output_tokens_per_s']-get('Output Token Throughput'))<1e-8
            for key,name in [('ttft','TTFT'),('tpot','TPOT'),('e2el','E2EL')]:
                assert m[key+'_mean_ms']==float(stats[name]['Average'].split()[0])
                assert m[key+'_p99_ms']==float(stats[name]['P99'].split()[0])
            formal[(phase['generation'],phase['input_tokens'])]=phase
    comparison=[]
    for length in [32768,65536,131072,262144]:
        sg=formal[('sglang-g132-original' if length<131072 else 'sglang-g133-confirm',length)]
        native=formal[('vllm-g134-current',length)]
        comparison.append({'input_tokens':length,'sglang_round':sg['round_id'],'vllm_round':native['round_id'],
            'sglang_output_tokens_per_s':sg['output_tokens_per_s'],'vllm_output_tokens_per_s':native['output_tokens_per_s'],
            'throughput_change_percent':(sg['output_tokens_per_s']/native['output_tokens_per_s']-1)*100,
            'latency_change_percent':{k:(v/native['metrics'][k]-1)*100 for k,v in sg['metrics'].items() if k.endswith('_ms')}})
    assert comparison==load(ROOT/'comparison.json')
    print(json.dumps({'status':'passed','scope':'离线证据复算，不是新设备实验或全模型精度验收',
        'phases':len(phases),'requests':count,'formal_phases':len(formal),'comparison':comparison},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
