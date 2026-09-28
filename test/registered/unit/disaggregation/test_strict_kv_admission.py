from types import SimpleNamespace

from sglang.srt.disaggregation.strict_kv_admission import projected_full_kv_tokens


def request(prompt: int, max_output: int, emitted: int = 0):
    return SimpleNamespace(
        origin_input_ids=range(prompt),
        output_ids=range(emitted),
        sampling_params=SimpleNamespace(max_new_tokens=max_output),
    )


def test_rejects_transfer_that_would_starve_running_output():
    running = request(31_104, 5_984, emitted=5_755)
    transfer = request(102_400, 256)
    projected = projected_full_kv_tokens(
        [running, transfer], page_size=64, decode_headroom_per_request=512
    )
    assert projected > 139_264


def test_admits_two_long_requests_with_sufficient_final_kv():
    requests = [request(64_000, 1_000), request(64_000, 1_000)]
    projected = projected_full_kv_tokens(
        requests, page_size=64, decode_headroom_per_request=512
    )
    assert projected <= 139_264


def test_deduplicates_a_request_seen_in_multiple_queue_snapshots():
    req = request(31_104, 5_984)
    once = projected_full_kv_tokens(
        [req], page_size=64, decode_headroom_per_request=512
    )
    twice = projected_full_kv_tokens(
        [req, req], page_size=64, decode_headroom_per_request=512
    )
    assert once == twice
