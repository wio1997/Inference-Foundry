"""Route-aware admission hints; native request bodies remain untouched.

Hints reserve intended output work while the HTTP lease is live. They neither
count committed tokens nor model native background work after HTTP completion.
"""
import json
from placement import estimate_request as legacy_estimate_request

def estimate_request(body,path=None):
    if path!="/v1/responses":
        return legacy_estimate_request(body)
    try:
        request=json.loads(body)
        budget=request.get("max_output_tokens",16)
        if type(budget)is not int or budget<=0:
            raise ValueError("unknown or invalid output budget")
        return budget,len(body)
    except (ValueError,TypeError,AttributeError):
        return 16,len(body)
