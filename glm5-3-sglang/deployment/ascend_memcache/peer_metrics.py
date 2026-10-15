#!/usr/bin/env python3
"""Read-only exporter for the physical peer's Prometheus mmap files."""
import argparse,re
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from prometheus_client import CollectorRegistry,generate_latest,multiprocess
ap=argparse.ArgumentParser();ap.add_argument('--log',required=True);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=31490);a=ap.parse_args()
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/health':code,body=200,b'OK'
  elif self.path=='/metrics':
   try:
    log=Path(a.log).read_text(errors='replace');dirs=re.findall(r'PROMETHEUS_MULTIPROC_DIR = (\S+)',log);assert dirs,'No live model metrics directory'
    r=CollectorRegistry();multiprocess.MultiProcessCollector(r,path=dirs[-1]);body=generate_latest(r);code=200
   except Exception as e:code,body=503,str(e).encode()
  else:code,body=404,b'Not found'
  self.send_response(code);self.send_header('Content-Type','text/plain; version=0.0.4');self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
ThreadingHTTPServer((a.host,a.port),H).serve_forever()
