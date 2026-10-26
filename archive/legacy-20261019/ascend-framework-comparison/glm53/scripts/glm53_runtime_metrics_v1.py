"""Bounded complete native metrics capture; per-DP running counts are sampled."""
import asyncio,gzip,hashlib,json,math,re,time
from pathlib import Path

CAP=8*1024**2

def decode_occupancy(framework, node, text):
    names=({'sglang:num_running_reqs':'running','sglang:num_queue_reqs':'waiting'}
           if framework=='sglang' else
           {'vllm:num_requests_running':'running','vllm:num_requests_waiting':'waiting'})
    result={}
    for line in text.splitlines():
        if line.startswith('#'):continue
        match=re.fullmatch(r'([^ {]+)\{(.*)\}\s+(\S+)(?:\s+\S+)?',line)
        if not match or match.group(1) not in names:continue
        labels=match.group(2);rank_label='dp_rank' if framework=='sglang' else 'engine'
        rank=re.search(r'(?:^|,)'+rank_label+r'="(\d+)"(?:,|$)',labels)
        assert rank is not None, 'Native per-DP label missing'
        if framework=='sglang':assert 'engine_type="decode"' in labels
        dp=int(rank.group(1));assert 0<=dp<16
        value=float(match.group(3));assert math.isfinite(value) and value>=0 and value.is_integer()
        kind=names[match.group(1)];row=result.setdefault(dp,{'node':node,'native_dp':dp})
        if kind in row:assert row[kind]==int(value),'Duplicate native gauges disagree'
        row[kind]=int(value)
    return result

class MetricRecorder:
    def __init__(self,framework,endpoints,output):
        assert framework in ['sglang','vllm']
        assert len(endpoints)==2 and all(e['role']=='D' for e in endpoints)
        self.framework=framework;self.endpoints=endpoints;self.output=Path(output)
        self.file=self.output.open('xb');self.gz=gzip.GzipFile(fileobj=self.file,mode='wb',mtime=0)
        self.samples=[]
    async def _fetch(self,session,endpoint):
        row={'node':endpoint['node'],'url':endpoint['url'],'dispatch_perf_counter_ns':time.perf_counter_ns()}
        try:
            async with session.get(endpoint['url'],timeout=5) as response:
                row['http_status']=response.status;raw=bytearray()
                async for block in response.content.iter_chunked(65536):
                    raw.extend(block)
                    if len(raw)>CAP:raise ValueError('Native metrics response exceeds8MiB cap; do not truncate')
                raw=bytes(raw);text=raw.decode();row.update(raw_utf8=text,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                assert response.status==200
                row['occupancy']=decode_occupancy(self.framework,endpoint['node'],text)
                assert row['occupancy'] and all('running' in x and 'waiting' in x for x in row['occupancy'].values())
        except Exception as error:row['error']=repr(error)
        row['terminal_perf_counter_ns']=time.perf_counter_ns()
        self.gz.write((json.dumps(row,ensure_ascii=False)+'\n').encode());self.gz.flush()
        return {k:v for k,v in row.items() if k!='raw_utf8'}
    async def sample(self,session):
        native=await asyncio.gather(*(self._fetch(session,e) for e in self.endpoints));combined={};errors=[]
        for row in native:
            if 'error' in row:errors.append({'node':row['node'],'error':row['error']});continue
            for dp,value in row['occupancy'].items():
                assert dp not in combined,'Native global DP reported by two API nodes'
                combined[dp]=value
        complete=not errors and set(combined)==set(range(16))
        row={'native':native,'complete_all_D_DP':complete,'per_DP':combined,'errors':errors}
        if complete:
            row['sampled_total_running']=sum(x['running'] for x in combined.values())
            row['sampled_total_waiting']=sum(x['waiting'] for x in combined.values())
            row['endpoint_terminal_skew_seconds']=(max(x['terminal_perf_counter_ns'] for x in native)-min(x['terminal_perf_counter_ns'] for x in native))/1e9
        self.samples.append(row);return row
    async def run(self,session,stop,interval=.5):
        while not stop.is_set():
            try:await asyncio.wait_for(stop.wait(),interval)
            except asyncio.TimeoutError:await self.sample(session)
    def close(self):
        self.gz.close();self.file.close()
        complete=[s for s in self.samples if s['complete_all_D_DP']]
        return {'sample_count':len(self.samples),'complete_sample_count':len(complete),
                'sampled_peak_total_D_running':max((s['sampled_total_running'] for s in complete),default=None),
                'sampled_peak_D_running_by_DP':{str(dp):max((s['per_DP'][dp]['running'] for s in complete),default=None) for dp in range(16)},
                'raw_archive':str(self.output),'raw_archive_sha256':hashlib.sha256(self.output.read_bytes()).hexdigest(),
                'samples':self.samples,'limitation':'Polling yields sampled occupancy, not an exact instantaneous peak. All request clocks remain in the separate full-run denominator.'}
