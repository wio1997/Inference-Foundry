"""Local source preparation only; no resource action or SDK compilation."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'runs/20261017T0740-sglang-config-source-v1/code';OUT=ROOT/'config/overlays/sglang-sdma-v1';OUT.mkdir(parents=True,exist_ok=False)
index=json.loads((SOURCE/'index.json').read_text());entries=[]
def load(local):
 entry=next(x for x in index if x['local']==local);raw=(SOURCE/local).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256'];return entry,raw.decode()
def replace(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
e,s=load('49-transfer_engine.py')
s=replace(s,'import logging','import logging\nimport glm53_sglang_sdma_audit as _glm53')
s=replace(s,'        self.engine = TransferEngine()','        _glm53.prepare_sdk()\n        self.engine = TransferEngine()')
s=replace(s,'        self.initialize()','        self.initialize()')
s=replace(s,'            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()','            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()\n        _glm53.initialized(self)')
s=replace(s,'        transfer_protocol = self._get_transfer_protocol()','        transfer_protocol = self._get_transfer_protocol()\n        if transfer_protocol != "sdma":\n            raise RuntimeError("GLM53 requires SDMA, no transport fallback")')
s=replace(s,'        if ret_value != 0:\n            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")','        _glm53.registered(self, ptrs, lengths, ret_value)')
s+='''
    def transfer_sync(self, session_id, buffer, peer_buffer_address, length):
        return _glm53.transfer(self, session_id, [buffer], [peer_buffer_address], [length], lambda: super(AscendTransferEngine, self).transfer_sync(session_id, buffer, peer_buffer_address, length))

    def batch_transfer_sync(self, session_id, buffers, peer_buffer_addresses, lengths):
        return _glm53.transfer(self, session_id, buffers, peer_buffer_addresses, lengths, lambda: super(AscendTransferEngine, self).batch_transfer_sync(session_id, buffers, peer_buffer_addresses, lengths))
'''
def save(entry,text):
 ast.parse(text);local=entry['path'].split('/python/',1)[1];p=OUT/'source'/local;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);entries.append({'container_path':entry['path'],'local':'source/'+local,'original_sha256':entry['sha256'],'patched_sha256':hashlib.sha256(text.encode()).hexdigest()})
save(e,s)
e,s=load('48-conn.py');s=replace(s,'import concurrent.futures','import concurrent.futures\nimport glm53_sglang_sdma_audit as _glm53')
s=replace(s,'            self.engine.batch_register(ptrs, lens)','            self.engine.batch_register(ptrs, lens)\n        _glm53.kv_registered(self)')
s=replace(s,'    def send_kvcache(\n','    @_glm53.send_audited\n    def send_kvcache(\n')
a=s.index('\n    def requires_dcp_relayout(')
s=s[:a]+'''
    @_glm53.send_audited
    def send_aux(self, req, prefill_aux_index, dst_aux_ptrs):
        return super().send_aux(req, prefill_aux_index, dst_aux_ptrs)

    def send_aux_tcp(self, *args, **kwargs):
        return _glm53.reject_tcp(*args, **kwargs)

    def send_aux_data_to_endpoint(self, *args, **kwargs):
        return _glm53.reject_tcp(*args, **kwargs)

    def _handle_aux_data(self, *args, **kwargs):
        return _glm53.reject_tcp(*args, **kwargs)
'''+s[a:];save(e,s)
helper=(ROOT/'scripts/glm53_sglang_sdma_audit.py').read_bytes();(OUT/'python').mkdir();(OUT/'python/glm53_sglang_sdma_audit.py').write_bytes(helper)
manifest={'project_id':'glm53','image_id':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','functional_only':True,'files':entries,'helpers':[{'local':'python/glm53_sglang_sdma_audit.py','sha256':hashlib.sha256(helper).hexdigest()}]};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
