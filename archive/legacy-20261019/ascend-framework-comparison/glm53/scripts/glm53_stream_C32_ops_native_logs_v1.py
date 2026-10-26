"""Archive exact native logs with low compression cost; preserve complete bytes/counters and an8GiB hard cap."""
import collections, gzip, hashlib, re

def summarize_log_stream(source, destination, max_bytes=8*1024**3):
    digest=hashlib.sha256();total=0;tail=collections.deque(maxlen=180)
    metadata=[];audits=[];capacity=collections.deque(maxlen=2000);diagnostics=collections.deque(maxlen=1000)
    errors=collections.deque(maxlen=100);error_count=0;capacity_count=0;selected_count=0
    sdk_tail=collections.deque(maxlen=10);sdk_lines=0;calls=0;copy_bytes=0;statuses=set()
    with destination.open('xb') as file:
        with gzip.GzipFile(fileobj=file,mode='wb',mtime=0,compresslevel=1) as archive:
            while True:
                raw=source.readline(8*1024**2+1)
                if not raw:break
                assert len(raw)<=8*1024**2,'Native log line exceeded8MiB; preserve capture as incomplete'
                total+=len(raw);assert total<=max_bytes,'Native full log exceeded capture cap; never silently truncate'
                digest.update(raw);archive.write(raw)
                line=raw.decode(errors='replace').rstrip('\r\n');tail.append(line)
                if 'GLM53_C32_OPS_' in line or 'GLM53_DEEPEP_METADATA_' in line or 'GLM53_METADATA_HOOK_FATAL' in line:
                    metadata.append(line);assert len(metadata)<=8192,'Unexpected unbounded metadata markers'
                if line.startswith(('(Worker','(EngineCore','(APIServer')) or any(key in line for key in ['GLM53_','ERROR','Traceback','Exception','Error:']):selected_count+=1
                if any(marker in line for marker in ['GLM53_SGLANG_SDMA=','GLM53_SGLANG_ENTRY=','GLM53_PD_SDMA=','GLM53_LIFECYCLE=','GLM53_MC_GET=','GLM53_SERVE_ENTRY=']):audits.append(line)
                if any(key in line for key in ['Load weight','Load weights','KV Cache','KV cache','memory pool','available_gpu_mem','mem_fraction_static','max_total_num_tokens','Attention TP','DP','hicache','NPU KV cache size:','Available KV cache memory:','Loading model weights took','Registered Mooncake KV caches:','assigned as DP rank','Capturing','capturing','Graph','graph','spec_decode/base']):
                    capacity.append(line);capacity_count+=1
                if any(key in line.lower() for key in ['backup','prefetch','host pool','storage hit','batch_put','batch_get']):diagnostics.append(line)
                if any(key in line for key in ['ERROR','Traceback','Exception','Error:']):errors.append(line);error_count+=1
                if 'hybm_data_op_sdma.cpp' in line:
                    sdk_lines+=1;sdk_tail.append(line)
                    match=re.search(r'CopyG2GAsync.*length:(\d+) st:(\d+)',line)
                    if match:calls+=1;copy_bytes+=int(match.group(1));statuses.add(int(match.group(2)))
    archive_digest=hashlib.sha256()
    with destination.open('rb') as file:
        for block in iter(lambda:file.read(1024**2),b''):archive_digest.update(block)
    return {'archive':str(destination),'archive_sha256':archive_digest.hexdigest(),
            'sha256':digest.hexdigest(),'bytes':total,'tail':'\n'.join(tail),
            'audit_lines':audits,'metadata_marker_lines':metadata,'selected_lines':selected_count,'capacity_lines':list(capacity),
            'capacity_selected_total':capacity_count,'cache_diagnostic_lines':list(diagnostics),
            'error_lines':list(errors),'error_line_count':error_count,'full_archive_complete':True,
            'native_sdma':{'source_lines':sdk_lines,'copy_calls':calls,'copy_descriptor_bytes':copy_bytes,
                           'copy_status_values':sorted(statuses),'physical_link_bytes':None,'last_lines':list(sdk_tail)}}
