# Extracted unchanged CPU methods; provenance in fixture-source.json.
from __future__ import annotations
class DecodeRankPaths:
    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        if prefill_info is None:
            return None
        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank
        if prefill_info.dp_size == 1:
            return 0
        if prefill_info.follow_bootstrap_room and (not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()):
            return req.bootstrap_room % prefill_info.dp_size
        return None
    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            self._cancel_prefill_dp_rank_queries()
            return
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)
        (ready_addrs, remaining) = self._ensure_prefill_info(addr_to_reqs)
        resolved: List[Tuple[DecodeRequest, int]] = []
        for (bootstrap_addr, decode_reqs) in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                prefetched = self._prefill_dp_rank_queries.pop(bootstrap_addr, None)
                prefetched_rooms = prefetched[0] if prefetched is not None else ()
                if prefetched is not None and tuple(rooms[:len(prefetched_rooms)]) == prefetched_rooms:
                    room_to_rank = prefetched[1].result()
                    remaining_rooms = rooms[len(prefetched_rooms):]
                    if remaining_rooms:
                        room_to_rank.update(CommonKVReceiver.query_prefill_dp_ranks(bootstrap_addr, remaining_rooms))
                else:
                    if prefetched is not None:
                        prefetched[1].cancel()
                    room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(bootstrap_addr, rooms)
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(str(decode_req.req.bootstrap_room))
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)
            else:
                prefetched = self._prefill_dp_rank_queries.pop(bootstrap_addr, None)
                if prefetched is not None:
                    prefetched[1].cancel()
        self.pending_reqs = remaining
        for (decode_req, prefill_dp_rank) in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)
            _pd_mon.prefill_resolved(decode_req.req, prefill_dp_rank)
