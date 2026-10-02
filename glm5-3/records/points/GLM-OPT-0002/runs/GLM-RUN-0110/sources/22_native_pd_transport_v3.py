"""Native PD V2 request contract with geometry-aware native D fallback."""
from native_pd_transport_v2 import NativePDTransport as _NativePDTransport
from native_pd_geometry import assess_pd_geometry,_positive
class NativePDTransport(_NativePDTransport):
    def __init__(self,producers,*args,native_plans=None,**kwargs):
        super().__init__(producers,*args,**kwargs)
        self.pd_geometry={}
        for origin,producer in self.producers.items():
            entry=(native_plans or{}).get(origin)
            if not isinstance(entry,dict):
                self.pd_geometry[origin]=dict(eligible=False,reasons=["native_pair_plans_unknown"]);continue
            result=assess_pd_geometry(entry.get("producer"),entry.get("decoder"),entry.get("connector_source_sha256"))
            if result["eligible"]:
                p=result["producer"];d=result["decoder"]
                try:
                    matching=(d["origin"]==origin and p["origin"]==producer["url"].rstrip("/")and p["host"]==producer["remote_host"]and _positive(p["kv"].get("kv_port"))==producer["remote_port"]and p["sizes"]["dcp_size"]==producer["dcp_size"]and p["sizes"]["pcp_size"]==producer.get("pcp_size",1))
                except (TypeError,ValueError):matching=False
                if not matching:
                    result["eligible"]=False;result["reasons"].append("native_pair_mapping_mismatch")
            self.pd_geometry[origin]=result
    async def handle_async_request(self,request):
        origin=str(request.url.copy_with(path="/",query=None,fragment=None)).rstrip("/")
        if origin in self.producers and request.method=="POST":
            geometry=self.pd_geometry[origin]
            if not geometry["eligible"]:
                self.trace("pd_geometry_native_fallback",decoder=origin,reasons=geometry["reasons"],internal_output_tokens=0)
                return await self.base.handle_async_request(request)
        return await super().handle_async_request(request)
