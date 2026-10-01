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
    def __init__(self,max_bytes=65536):
        self.max_bytes=max_bytes;self.line=bytearray();self.data=[];self.size=0
        self.drop_line=False;self.drop_frame=False;self.failed=False;self.errors=0;self.output_started=False
    def feed(self,block):
        previous=self.errors
        pieces=block.split(b"\n")
        for index,piece in enumerate(pieces):
            end=index<len(pieces)-1
            if not self.drop_line:
                if len(self.line)+len(piece)>self.max_bytes:
                    self.line.clear();self.drop_line=True;self.drop_frame=True
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
                if not self.output_started or b'"error"' in value:
                    try:event=json.loads(value)
                    except (ValueError,UnicodeDecodeError):event=None
                    error=native_error(event)
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
            if self.size>self.max_bytes:self.data=[];self.drop_frame=True
            else:self.data.append(value)
