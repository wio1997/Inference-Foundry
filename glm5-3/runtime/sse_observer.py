"""Bounded native SSE error observation; callers forward original bytes unchanged."""
import json

def native_error(event):
    if isinstance(event,dict) and isinstance(event.get("error"),dict):
        return event["error"]
    return None

def server_error(error):
    if not isinstance(error,dict):return False
    try:
        if int(error.get("code",0))>=500:return True
    except (TypeError,ValueError):pass
    return error.get("type") in ("InternalServerError","EngineDeadError")

class NativeSSEObserver:
    """Observe complete uncompressed frames, including split CRLF and multiline data.

    Oversized frames are discarded from observation only; all wire bytes still
    pass through. This is fault telemetry, not token counting or success proof.
    """
    def __init__(self,max_bytes=65536,collect_contract=False):
        self.collect_contract=collect_contract;self.done=False;self.usage=None;self.finish_reasons={};self.contract_error=False;self.contract_unknown=False
        self.max_bytes=max_bytes;self.line=bytearray();self.data=[];self.size=0
        self.drop_line=False;self.drop_frame=False;self.failed=False;self.errors=0;self.output_started=False
    def feed(self,block):
        previous=self.errors
        pieces=block.split(b"\n")
        for index,piece in enumerate(pieces):
            end=index<len(pieces)-1
            if not self.drop_line:
                if len(self.line)+len(piece)>self.max_bytes:
                    self.line.clear();self.drop_line=True;self.drop_frame=True;self.contract_unknown=True
                else:self.line.extend(piece)
            if end:
                line=bytes(self.line).removesuffix(b"\r")
                if not self.drop_line:self._line(line)
                self.line.clear();self.drop_line=False
        return self.errors>previous
    def _line(self,line):
        if not line:
            if not self.drop_frame and self.data:
                value=b"\n".join(self.data)
                if self.collect_contract and value==b'[DONE]':self.done=True
                elif self.collect_contract and self.done:self.contract_unknown=True
                if value!=b'[DONE]' and (self.collect_contract or not self.output_started or b'"error"' in value):
                    try:event=json.loads(value)
                    except (ValueError,UnicodeDecodeError):event=None
                    error=native_error(event)
                    if self.collect_contract:
                        if not isinstance(event,dict):self.contract_unknown=True
                        if error is not None:self.contract_error=True
                        if isinstance(event,dict):
                            if isinstance(event.get('usage'),dict):self.usage=event['usage']
                            for choice in event.get('choices') if isinstance(event.get('choices'),list) else []:
                                if isinstance(choice,dict) and choice.get('finish_reason') is not None:self.finish_reasons[str(choice.get('index',0))]=choice['finish_reason']
                    if server_error(error):self.failed=True;self.errors+=1
                    if isinstance(event,dict) and error is None:
                        for choice in (event.get("choices") if isinstance(event.get("choices"),list) else []):
                            if not isinstance(choice,dict):continue
                            delta=choice.get("delta") or {}
                            if not isinstance(delta,dict):continue
                            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or delta.get("tool_calls") or delta.get("function_call") or choice.get("text"):
                                self.output_started=True
            self.data=[];self.size=0;self.drop_frame=False;return
        if self.drop_frame:return
        if line.startswith(b"data:"):
            value=line[5:].removeprefix(b" ");self.size+=len(value)
            if self.size>self.max_bytes:self.data=[];self.drop_frame=True;self.contract_unknown=True
            else:self.data.append(value)

    def contract(self):
        return {'done':self.done,'usage':self.usage,'finish_reasons':dict(self.finish_reasons),'native_error':self.contract_error,'unknown':self.contract_unknown}
