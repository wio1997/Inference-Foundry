"""Task-only read-only mTLS file server; no directory listing or write API."""
import hashlib,http.server,json,os,pathlib,ssl,threading,time,urllib.parse
ROOT=pathlib.Path('/work');MODEL=pathlib.Path('/model');lock=threading.Lock();rate_lock=threading.Lock();next_read=0.0
manifest=json.loads((ROOT/'initial-manifest.json').read_text());state={'status':'SERVING','files':manifest['files'],'errors':[],'bytes_sent':0,'active_requests':0};allowed={44:'172.16.12.44',46:'172.16.12.46'}
def save():
 p=ROOT/'server-state.json.tmp';p.write_text(json.dumps(state));os.replace(p,ROOT/'server-state.json')
def throttle(count):
 global next_read
 with rate_lock:
  now=time.monotonic();start=max(now,next_read);next_read=start+count/(512*1024**2)
 delay=start-time.monotonic()
 if delay>0:time.sleep(delay)
def snapshot():
 with lock:return json.loads(json.dumps(state))
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):
  try:
   peer=self.connection.getpeercert();cn=next(v for attrs in peer['subject'] for k,v in attrs if k=='commonName')
   assert any(cn=='mech36-node'+str(n) and self.client_address[0]==ip for n,ip in allowed.items()),'Unrecognized transfer client'
   path=urllib.parse.urlsplit(self.path).path
   if path=='/manifest':
    data=json.dumps(snapshot()).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
   assert path.startswith('/file/'),'Unknown endpoint'
   name=urllib.parse.unquote(path[len('/file/'):]);assert name in state['files'] and '/' not in name and '\\' not in name,'Unlisted file'
   item=state['files'][name];file=(ROOT/'image-c5.tar') if name=='image-c5.tar' else MODEL/name
   before=file.stat();assert [before.st_size,before.st_mtime_ns,before.st_ino]==item['stat'],'Source changed before read'
   with lock:state['active_requests']+=1
   digest=hashlib.sha256();self.send_response(200);self.send_header('Content-Length',str(before.st_size));self.end_headers()
   try:
    with file.open('rb') as f:
     while True:
      throttle(4*1024**2);block=f.read(4*1024**2)
      if not block:break
      digest.update(block);self.wfile.write(block)
      with lock:state['bytes_sent']+=len(block)
    after=file.stat();assert [after.st_size,after.st_mtime_ns,after.st_ino]==item['stat'],'Source changed during read'
    sha=digest.hexdigest()
    with lock:
     assert item['sha256'] in [None,sha],'Content changed between readers'
     item['sha256']=sha;item['completed_streams']=item.get('completed_streams',0)+1;save()
   finally:
    with lock:state['active_requests']-=1
  except Exception as e:
   with lock:state['errors'].append({'type':type(e).__name__,'message':str(e)});save()
   self.close_connection=True
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
context.load_cert_chain(ROOT/'secrets/server.crt',ROOT/'secrets/server.key');context.load_verify_locations(ROOT/'secrets/ca.crt');context.verify_mode=ssl.CERT_REQUIRED
server=http.server.ThreadingHTTPServer(('172.16.12.37',39039),Handler);server.daemon_threads=True
server.socket=context.wrap_socket(server.socket,server_side=True)
with lock:save()
server.serve_forever()
