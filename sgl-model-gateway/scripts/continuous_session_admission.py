#!/usr/bin/env python3
"""Replenish bounded session-admission credits through an existing gateway API.

CPU-only, Python 3.7+, standard library. Never starts/restarts workers, changes
session pins, or enables a paused gateway. Core response leases and minimum
assignment intervals remain authoritative. Operator pause must share ops_lock.
"""
import argparse
import fcntl
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import threading
import time
from urllib.parse import urlencode, urlsplit

MIGRATION = '/workers/session-handoff/migration'
HANDOFF = '/workers/session-handoff'
MAX_SWITCHES = 65536
DEFAULT_POLICY = dict(interval_secs=60, cold_limit=1, recent_secs=900,
                      max_body_bytes=1048576, cohort_size=4,
                      max_new_pressure=3, max_new_waiting=1,
                      relative_slack=1, new_sessions_bps=10000)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def select_sessions(migration, private_hashes, policy):
    rows = []
    for entry in migration['sessions']:
        idle = entry.get('idle_seconds')
        size = entry.get('last_body_bytes')
        key = entry.get('session_hash', '')
        if (entry.get('destination') == 'legacy' and key not in private_hashes
                and re.fullmatch('[a-f0-9]{64}', key)
                and entry.get('last_completed_cleanly') is True
                and finite(idle) and 0 <= idle <= policy['recent_secs']
                and type(size) is int and 0 < size <= policy['max_body_bytes']):
            rows.append(entry)
    rows.sort(key=lambda x: (x['idle_seconds'] > 120,
                             x['last_body_bytes'], x['idle_seconds']))
    return [x['session_hash'] for x in rows[:policy['cohort_size']]]


def plan(migration, handoff, workers, feedback, private_hashes, policy,
         error_baseline, error_latched=False):
    """Pure policy; return desired config plus a fixed reason and audit fields."""
    current = migration['config']
    used = migration['enrolled'] + migration['migrated']
    if type(used) is not int or used < 0 or used > MAX_SWITCHES:
        raise ValueError('invalid_assignment_count')
    output = dict(reason='manual_paused', used=used, selected=[],
                  error_latched=error_latched, coverage='complete', config=None)
    if not current['enabled']:
        return output
    desired = dict(current)
    desired.update(new_sessions_bps=0, legacy_sessions_bps=0,
                   max_cold_inflight=policy['cold_limit'],
                   min_interval_secs=policy['interval_secs'], quiet_secs=0,
                   max_switches=used, session_hashes=[])
    output['config'] = desired
    if (handoff['persistence_failed'] or handoff['config']['candidate_bps'] != 0):
        output['reason'] = 'handoff_not_ready'
        return output
    if (workers['stats'] != dict(prefill_count=4, decode_count=8, regular_count=0)
            or len(workers['workers']) != 12
            or not all(x['is_healthy'] for x in workers['workers'])):
        output['reason'] = 'workers_not_ready'
        return output
    if not feedback.get('fresh'):
        output['reason'] = 'feedback_unavailable'
        return output
    errors = feedback.get('candidate_errors')
    if not finite(errors) or errors < error_baseline:
        output['reason'] = 'error_counter_unavailable'
        return output
    if errors > error_baseline:
        output['error_latched'] = True
    if output['error_latched']:
        output['reason'] = 'candidate_error_latched'
        return output
    for key in ['new_pressure', 'old_pressure', 'new_waiting']:
        if not finite(feedback.get(key)) or feedback[key] < 0:
            output['reason'] = 'invalid_load_feedback'
            return output
    if used >= MAX_SWITCHES:
        output['reason'] = 'ledger_lifetime_limit'
        return output
    if migration['cold_inflight'] > 0:
        output['reason'] = 'cold_session_inflight'
        return output
    if (feedback['new_pressure'] > policy['max_new_pressure']
            or feedback['new_waiting'] > policy['max_new_waiting']
            or feedback['new_pressure'] > feedback['old_pressure'] + policy['relative_slack']):
        output['reason'] = 'new_prefill_pressure'
        return output
    selected = select_sessions(migration, private_hashes, policy)
    if migration.get('tracked_sessions', len(migration['sessions'])) > len(migration['sessions']):
        output['coverage'] = 'recent_metadata_incomplete'
    desired.update(new_sessions_bps=policy['new_sessions_bps'],
                   max_switches=used + 1, session_hashes=selected)
    output.update(reason='credit_ready', selected=selected)
    return output


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def request(origin, method, path, value=None):
    origin = urlsplit(origin)
    if origin.scheme != 'http' or origin.username or origin.password:
        raise ValueError('http_origin_required')
    connection = http.client.HTTPConnection(origin.hostname, origin.port or 80, timeout=5)
    try:
        connection.request(method, path, body=None if value is None else json.dumps(value),
                           headers={'Content-Type': 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        raw = response.read(4 * 1024 * 1024 + 1)
        if response.status != 200 or len(raw) > 4 * 1024 * 1024:
            raise RuntimeError('http_status_' + str(response.status))
        return json.loads(raw)
    finally:
        connection.close()


class DockerHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect('/var/run/docker.sock')


def inspect_gateway(cid):
    connection = DockerHTTP('localhost')
    try:
        connection.request('GET', '/containers/' + cid + '/json')
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError('gateway_identity_unavailable')
        return json.loads(response.read(2 * 1024 * 1024))
    finally:
        connection.close()


def read_feedback(config):
    names = ('llm_pd:service_complete', 'llm_pd:prefill_running_by_service',
             'llm_pd_exporter_source_up', 'llm_pd_queue_requests', 'llm_gateway_requests_total')
    query = '{__name__=~"' + '|'.join(names) + '",cluster="' + config['cluster'] + '"}'
    result = request(config['prometheus'], 'GET', '/api/v1/query?' + urlencode({'query': query}))
    if result.get('status') != 'success':
        raise RuntimeError('prometheus_query_failed')
    complete, sources, queues, running = set(), set(), {}, {}
    errors = []
    now = time.time()
    invalid = False
    for row in result['data']['result']:
        labels = row['metric']
        name, service = labels.get('__name__'), labels.get('service_id')
        relevant = (name in ['llm_pd:service_complete', 'llm_pd_exporter_source_up',
                             'llm_pd:prefill_running_by_service']
                    or (name == 'llm_pd_queue_requests' and labels.get('role') == 'prefill')
                    or (name == 'llm_gateway_requests_total'
                        and labels.get('gateway_scope') == 'front'
                        and labels.get('outcome') == 'error'))
        if not relevant:
            continue
        value = float(row['value'][1])
        timestamp = float(row['value'][0])
        if (not finite(value) or value < 0 or not finite(timestamp)
                or not -2 <= now - timestamp <= config.get('max_feedback_age_secs', 30)):
            invalid = True
            continue
        if name == 'llm_pd:service_complete' and value == {
                'prefill-0': 2, 'prefill-1': 2, 'decode-0': 8}.get(service):
            complete.add(service)
        if name == 'llm_pd_exporter_source_up' and value == 1:
            sources.add(labels.get('node'))
        if name == 'llm_pd:prefill_running_by_service':
            running[service] = value
        if name == 'llm_pd_queue_requests' and labels.get('role') == 'prefill':
            key = (service, labels.get('node'), labels.get('dp_rank'), labels.get('stage'))
            queues[key] = max(value, queues.get(key, 0))
        if (name == 'llm_gateway_requests_total' and labels.get('gateway_scope') == 'front'
                and labels.get('outcome') == 'error'):
            errors.append(value)
    stages = {}
    for (service, _node, _rank, stage), value in queues.items():
        key = (service, stage)
        stages[key] = stages.get(key, 0) + value
    def pressure(service):
        return max(running.get(service, 0) + stages.get((service, 'waiting'), 0),
                   stages.get((service, 'bootstrap'), 0) + stages.get((service, 'prefill_inflight'), 0))
    required = {(entry['service_id'], entry['node'], str(entry['dp_rank']), stage)
                for entry in config['prefill_rank_locations']
                for stage in ['waiting', 'bootstrap', 'prefill_inflight']}
    fresh = (not invalid and len(config['prefill_rank_locations']) == 4
             and len(required) == 12 and required <= set(queues)
             and set(['prefill-0', 'prefill-1', 'decode-0']) <= complete
             and set(config['expected_nodes']) <= sources
             and set(['prefill-0', 'prefill-1']) <= set(running)
             and all((service, 'waiting') in stages for service in ['prefill-0', 'prefill-1']))
    return dict(fresh=fresh, candidate_errors=sum(errors) if errors else None,
                old_pressure=pressure('prefill-0'), new_pressure=pressure('prefill-1'),
                new_waiting=stages.get(('prefill-1', 'waiting'), 0))


def persist_status(config, result, previous):
    result = dict(result, observed_unix=time.time())
    result.setdefault('error_latched', bool(previous and previous.get('error_latched')))
    atomic_json(config['status_file'], result)
    if (previous is None or any(previous.get(k) != result.get(k)
                                for k in ['reason', 'used', 'selected', 'error_latched'])):
        # Fixed codes and counts only in journal; hashes stay in the private state file.
        print(json.dumps({k: result.get(k) for k in ['observed_unix', 'reason', 'used', 'error_latched']}), flush=True)
    return result


def tick(config, dry_run=False):
    previous = json.loads(Path(config['status_file']).read_text()) if Path(config['status_file']).exists() else {}
    try:
        feedback = read_feedback(config)
    except Exception:
        feedback = {'fresh': False}
    lock_path = Path(config['ops_lock'])
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'reason': 'operator_lock_busy'}
        info = inspect_gateway(config['gateway_cid'])
        if (info['Id'] != config['gateway_cid'] or not info['State']['Running']
                or info['State']['StartedAt'] != config['gateway_started_at']):
            result = dict(reason='gateway_generation_changed', config=None,
                          error_latched=previous.get('error_latched', False))
            return result if dry_run else persist_status(config, result, previous)
        origin = config['gateway']
        migration = request(origin, 'GET', MIGRATION)
        desired_path = Path(config['desired_file'])
        saved = json.loads(desired_path.read_text())
        if saved.get('controller_pending') is not None and not dry_run:
            # Reconcile an earlier ambiguous PUT before authorizing another one.
            saved.update(migration=migration['config'], updated_unix=time.time())
            saved.pop('controller_pending', None)
            atomic_json(desired_path, saved)
        handoff = request(origin, 'GET', HANDOFF)
        workers = request(origin, 'GET', '/workers')
        rollout = request(origin, 'GET', '/workers/prefill-rollout')
        if (rollout.get('new_prefix_bps') != 10000
                or rollout.get('candidate_new_only') is not True
                or rollout.get('incumbent_urls') != config['incumbent_urls']
                or set(w['url'] for w in workers['workers']) != set(config['expected_workers'])):
            feedback['fresh'] = False
        if feedback.get('fresh'):
            legacy = request(config['legacy_gateway'], 'GET', '/workers')
            old_local = sum(w['load'] for w in legacy['workers']
                            if w['worker_type'] == 'prefill')
            new_local = sum(w['load'] for w in workers['workers']
                            if w['worker_type'] == 'prefill' and w['url'].startswith(config['new_prefill_url'] + '@'))
            feedback['old_pressure'] = max(feedback['old_pressure'], old_local)
            feedback['new_pressure'] = max(feedback['new_pressure'], new_local)
        private = set(json.loads(Path(config['private_hashes_file']).read_text()))
        policy = dict(DEFAULT_POLICY, **config.get('policy', {}))
        result = plan(migration, handoff, workers, feedback, private, policy,
                      config['error_baseline'], previous.get('error_latched', False))
        result.update(feedback=feedback, gateway_started_at=config['gateway_started_at'])
        desired = result['config']
        if not dry_run and desired is not None and desired != migration['config']:
            # Supported operator tools share this lock. Recheck direct API changes too.
            if request(origin, 'GET', MIGRATION)['config'] != migration['config']:
                return persist_status(config, dict(reason='operator_config_changed', config=None), previous)
            saved = json.loads(desired_path.read_text())
            saved['controller_pending'] = dict(
                config=desired, used=result['used'],
                gateway_started_at=config['gateway_started_at'], prepared_unix=time.time())
            atomic_json(desired_path, saved)
            if request(origin, 'PUT', MIGRATION, desired) != desired:
                raise RuntimeError('configuration_ack_mismatch')
            if request(origin, 'GET', MIGRATION)['config'] != desired:
                raise RuntimeError('configuration_readback_mismatch')
        saved_config = desired if desired is not None else (
            migration['config'] if not migration['config']['enabled'] else None)
        if not dry_run and saved_config is not None:
            path = desired_path
            saved = json.loads(path.read_text())
            if saved.get('migration') != saved_config or saved.get('controller_pending') is not None:
                saved.update(migration=saved_config, updated_unix=time.time())
                saved.pop('controller_pending', None)
                atomic_json(path, saved)
        return result if dry_run else persist_status(config, result, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 10 <= config.get('poll_secs', 30) <= 60:
        raise ValueError('invalid_poll_interval')
    if not finite(config['error_baseline']) or config['error_baseline'] < 0:
        raise ValueError('invalid_error_baseline')
    singleton = Path(config['singleton_lock'])
    fd = os.open(str(singleton), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.once or args.dry_run:
            print(json.dumps(tick(config, dry_run=args.dry_run), indent=2))
            return
        stopping = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        signal.signal(signal.SIGINT, lambda *_: stopping.set())
        while not stopping.is_set():
            try:
                tick(config)
            except Exception as error:
                # Ambiguous PUTs are never retried here; next tick reads live counters.
                print(json.dumps({'reason': 'control_cycle_failed', 'type': type(error).__name__}), flush=True)
            stopping.wait(config.get('poll_secs', 30))


if __name__ == '__main__':
    main()
