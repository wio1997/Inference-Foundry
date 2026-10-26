"""Integrity-framed gzip transport for large read-only native JSON evidence."""
import base64,gzip,hashlib,io,json
MAX_RAW=256*1024**2
MAX_PACKED=8*1024**2
ENCODER=r'''
import base64
native_raw=json.dumps(out).encode();assert len(native_raw)<=256*1024**2
native_packed=gzip.compress(native_raw,mtime=0);assert len(native_packed)<=8*1024**2
native_archive=archive/'native-output.json.gz'
with native_archive.open('xb') as file:file.write(native_packed)
print('GLM53_NATIVE_GZIP='+json.dumps({'version':1,'raw_bytes':len(native_raw),'raw_sha256':hashlib.sha256(native_raw).hexdigest(),'packed_bytes':len(native_packed),'packed_sha256':hashlib.sha256(native_packed).hexdigest(),'archive':str(native_archive),'gzip_base64':base64.b64encode(native_packed).decode()}),flush=True)
'''
def decode(response):
 assert response['rc']==0
 lines=[line.split('=',1)[1] for line in response['stdout'].splitlines() if line.startswith('GLM53_NATIVE_GZIP=')];assert len(lines)==1,'Missing complete native integrity frame'
 frame=json.loads(lines[0]);assert frame['version']==1 and 0<frame['raw_bytes']<=MAX_RAW and 0<frame['packed_bytes']<=MAX_PACKED
 packed=base64.b64decode(frame['gzip_base64'],validate=True);assert len(packed)==frame['packed_bytes'] and hashlib.sha256(packed).hexdigest()==frame['packed_sha256']
 with gzip.GzipFile(fileobj=io.BytesIO(packed)) as file:raw=file.read(MAX_RAW+1)
 assert len(raw)==frame['raw_bytes']<=MAX_RAW and hashlib.sha256(raw).hexdigest()==frame['raw_sha256'];json.loads(raw)
 return response|{'stdout':raw.decode(),'native_output_transport':{k:v for k,v in frame.items() if k!='gzip_base64'}|{'wire_stdout_sha256':hashlib.sha256(response['stdout'].encode()).hexdigest()}}
