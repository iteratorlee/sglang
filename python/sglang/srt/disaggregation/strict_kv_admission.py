"""Conservative full-KV capacity projection for decode-side PD admission.

The ordinary preallocation budget clips estimated output lengths. That is
useful for throughput, but a transferred request cannot be retracted while a
running request grows. With long outputs the clipped budget can therefore
admit a transfer that consumes the running request's remaining KV capacity.
"""

from typing import Iterable


def projected_full_kv_tokens(
    requests: Iterable,
    *,
    page_size: int,
    decode_headroom_per_request: int,
) -> int:
    """Upper-bound page-aligned final KV occupancy for distinct live requests."""
    total = 0
    seen = set()
    for req in requests:
        if id(req) in seen:
            continue
        seen.add(id(req))
        prompt_len = len(req.origin_input_ids)
        output_len = max(req.sampling_params.max_new_tokens, len(req.output_ids))
        final_len = prompt_len + output_len + decode_headroom_per_request
        total += ((final_len + page_size - 1) // page_size) * page_size
    return total
