# SPDX-License-Identifier: Apache-2.0
"""Bounded raw HBM hashing for diagnostic SDMA payload identity, not performance.

ACL memcpy copies immediately without implicit stream/device synchronization:
https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0105.html
Use explicit synchronization and log this diagnostic ordering boundary.
HTTP carries addresses, byte counts and SHA only, never inference payload bytes.
"""
import ctypes
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PayloadHasher:
    def __init__(self, device_id, regions):
        self.device_id = int(device_id)
        self.regions = regions
        self.acl = ctypes.CDLL('libascendcl.so')
        self.acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
        self.acl.aclrtSetDevice.restype = ctypes.c_int32
        self.acl.aclrtSynchronizeDevice.argtypes = []
        self.acl.aclrtSynchronizeDevice.restype = ctypes.c_int32
        self.acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32]
        self.acl.aclrtMemcpy.restype = ctypes.c_int32
        self.lock = threading.Lock()
        self.sealed = False

    def seal(self):
        with self.lock:
            self.sealed = True

    def hash_ranges(self, addresses, sizes):
        if len(addresses) != len(sizes) or not 0 < len(sizes) <= 4096:
            raise ValueError('Invalid range count')
        if any(type(x) is not int or x <= 0 for x in addresses + sizes):
            raise ValueError('Positive integer addresses/sizes required')
        if sum(sizes) > 64 * 1024 * 1024:
            raise ValueError('Diagnostic byte bound exceeded')
        with self.lock:
            if self.sealed:
                raise RuntimeError('Owned payload hasher is sealed')
            for ptr, size in zip(addresses, sizes):
                if not any(base <= ptr and ptr + size <= base + length for base, length in self.regions):
                    raise ValueError('Hash range outside this engine registered memory')
            ret = self.acl.aclrtSetDevice(self.device_id)
            if ret != 0: raise RuntimeError('aclrtSetDevice returned ' + str(ret))
            ret = self.acl.aclrtSynchronizeDevice()
            if ret != 0: raise RuntimeError('aclrtSynchronizeDevice returned ' + str(ret))
            digest = hashlib.sha256()
            per_range = []
            for ptr, size in zip(addresses, sizes):
                host = ctypes.create_string_buffer(size)
                ret = self.acl.aclrtMemcpy(ctypes.addressof(host), size, ptr, size, 2)
                if ret != 0: raise RuntimeError('aclrtMemcpy D2H returned ' + str(ret))
                data = host.raw
                digest.update(data)
                per_range.append(hashlib.sha256(data).hexdigest())
        return {'sha256': digest.hexdigest(), 'range_sha256': per_range, 'bytes': sum(sizes), 'segments': len(sizes), 'device_id': self.device_id, 'explicit_device_sync': True}


def start_hash_server(hasher, host, port, allowed_peers):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            try:
                if self.path != '/hash' or self.client_address[0] not in allowed_peers:
                    raise ValueError('Unexpected peer/endpoint')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 256 * 1024: raise ValueError('Invalid control body size')
                request = json.loads(self.rfile.read(length))
                result = hasher.hash_ranges(request['addresses'], request['sizes'])
                body = json.dumps(result).encode()
                self.send_response(200)
            except Exception as exc:
                body = json.dumps({'error': repr(exc)}).encode()
                self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name='flash-payload-hash-control', daemon=True)
    server._flash_thread = thread
    thread.start()
    return server
