import json,urllib.request
with urllib.request.urlopen('http://172.16.12.32:29481/metrics',timeout=5) as f:s=f.read().decode()
x={z.split(' ',1)[0]:float(z.rsplit(' ',1)[1]) for z in s.splitlines() if z.startswith(('vllm:prefix_cache_hits_total{','vllm:prefix_cache_queries_total{'))}
assert x
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
