"""Conservative full-KV capacity projection for decode-side PD admission.

The ordinary preallocation budget clips estimated output lengths. That is
useful for throughput, but a transferred request cannot be retracted while a
running request grows. With long outputs the clipped budget can therefore
admit a transfer that consumes the running request's remaining KV capacity.
"""

from typing import Iterable


class StrictKVBypassGuard:
    """Bound admissions past the oldest FULL-KV-blocked decode request.

    Only successful admissions count as bypasses. Once the limit is reached,
    younger requests cannot consume capacity until the protected request is
    admitted or removed. The request identity is the DecodeRequest's identity,
    so duplicate request IDs or mutable Req fields cannot affect accounting.
    """

    def __init__(self, max_bypasses: int = 8):
        self.max_bypasses = max_bypasses
        self.blocked_entry = None
        self.bypasses = 0

    @property
    def blocked_id(self):
        return id(self.blocked_entry) if self.blocked_entry is not None else None

    @property
    def at_limit(self) -> bool:
        return self.blocked_entry is not None and self.bypasses >= self.max_bypasses

    def reconcile(self, live_entries: Iterable) -> None:
        if self.blocked_entry is not None and not any(
            entry is self.blocked_entry for entry in live_entries
        ):
            self.blocked_entry = None
            self.bypasses = 0

    def blocked(self, entry) -> None:
        if self.blocked_entry is None:
            self.blocked_entry = entry
            self.bypasses = 0

    def admitted(self, entry) -> None:
        if entry is self.blocked_entry:
            self.blocked_entry = None
            self.bypasses = 0
        elif self.blocked_entry is not None:
            self.bypasses += 1


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
