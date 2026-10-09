"""Cancellation races for /health_generate, using the production functions.

Load only the relevant function definitions from source because importing the
server pulls in GPU and optional tokenizer dependencies unavailable in CPU CI.
The generated request coroutine and its cleanup/abort methods are not copied.
"""

import ast
import asyncio
import copy
import logging
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
HTTP_SERVER = ROOT / "python/sglang/srt/entrypoints/http_server.py"
TOKENIZER_MANAGER = ROOT / "python/sglang/srt/managers/tokenizer_manager.py"
MANAGER_UTILS = ROOT / "python/sglang/srt/managers/utils.py"
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"
HEALTH_PREFIX = "HEALTH_CHECK"


def load_function(path, name, namespace, cls=None):
    """Compile an unmodified production function with small dependency stubs."""
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if cls is not None:
        nodes = next(
            node.body
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == cls
        )
    func = copy.deepcopy(
        next(
            node
            for node in nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
    )
    func.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                func,
            ],
            type_ignores=[],
        )
    )
    scope = dict(namespace)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class GenerateReqInput:
    def __init__(self, rid, **kwargs):
        self.rid = rid
        self.max_thinking_tokens = None
        self.routed_dp_rank = None
        self.is_single = True
        self.return_prompt_token_ids = False

    def normalize_batch_and_arguments(self):
        pass


class EmbeddingReqInput(GenerateReqInput):
    pass


class AbortReq:
    def __init__(self, rid, abort_all=False):
        self.rid = rid
        self.abort_all = abort_all


def build_manager_class():
    namespace = {
        "asyncio": asyncio,
        "BaseException": BaseException,
        "GenerateReqInput": GenerateReqInput,
        "AbortReq": AbortReq,
        "get_disagg": lambda: SimpleNamespace(language_only=False),
        "get_serving": lambda: SimpleNamespace(
            enable_strict_thinking=False, tokenizer_worker_num=1
        ),
        "HEALTH_CHECK_RID_PREFIX": HEALTH_PREFIX,
        "logger": logging.getLogger(__name__),
    }
    methods = {}
    for name in (
        "generate_request",
        "_release_req_states_on_failure",
        "abort_request",
        "_handle_abort_req",
    ):
        methods[name] = load_function(
            TOKENIZER_MANAGER, name, namespace, "TokenizerManager"
        )
    return type("Manager", (), methods)


Manager = build_manager_class()


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now


class HealthGenerateCleanupTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager()
        self.manager.rid_to_state = {}
        self.manager.encoder_dispatch_ready = {}
        self.manager.is_pause_cond = asyncio.Condition()
        self.manager.is_pause = False
        self.manager.request_logger = SimpleNamespace(
            log_received_request=lambda *args: None
        )
        self.manager.tokenizer = None
        self.manager.last_receive_tstamp = 0.0
        self.manager.server_status = "Up"
        self.manager.gracefully_exit = False
        self.manager.is_generation = True
        self.manager.enable_metrics = False
        self.manager.dispatched = asyncio.Event()
        self.manager.send_entered = asyncio.Event()
        self.manager.allow_dispatch = asyncio.Event()
        self.manager.allow_dispatch.set()
        self.manager.sent = []
        self.manager.auto_create_handle_loop = lambda: None
        self.manager._set_default_priority = lambda obj: None
        self.manager._dispatch_to_scheduler = self.manager.sent.append

        @asynccontextmanager
        async def reader_lock():
            yield

        self.manager.model_update_lock = SimpleNamespace(reader_lock=reader_lock())

        async def validate(obj):
            pass

        self.manager._validate_and_resolve_lora = validate

        def init_req_state(obj, request):
            self.manager.rid_to_state[obj.rid] = SimpleNamespace(
                dispatched=False, abort_sent=False
            )

        self.manager._init_req_state = init_req_state

        async def tokenize(obj):
            return SimpleNamespace(input_ids=[0])

        self.manager._tokenize_one_request = tokenize

        async def send(obj):
            self.manager.send_entered.set()
            await self.manager.allow_dispatch.wait()
            self.manager.rid_to_state[self.probe_rid].dispatched = True
            self.manager.dispatched.set()

        self.manager._send_one_request = send

        async def wait_response(obj, request):
            await asyncio.Event().wait()
            yield {}  # make this an async iterator

        self.manager._wait_one_response = wait_response
        self.clock = Clock()

    def make_endpoint(self, sleep, timeout=2):
        namespace = {
            "asyncio": SimpleNamespace(create_task=asyncio.create_task, sleep=sleep),
            "_global_state": SimpleNamespace(tokenizer_manager=self.manager),
            "ServerStatus": SimpleNamespace(
                Starting="Starting", Up="Up", UnHealthy="UnHealthy"
            ),
            "envs": SimpleNamespace(
                SGLANG_DIAG_BYPASS_HEALTH_GENERATE=SimpleNamespace(get=lambda: False),
                SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=SimpleNamespace(
                    get=lambda: True
                ),
            ),
            "GenerateReqInput": GenerateReqInput,
            "EmbeddingReqInput": EmbeddingReqInput,
            "get_disagg": lambda: SimpleNamespace(disaggregation_mode="none"),
            "DisaggregationMode": SimpleNamespace(NULL=SimpleNamespace(value="none")),
            "FAKE_BOOTSTRAP_HOST": "fake",
            "HEALTH_CHECK_RID_PREFIX": HEALTH_PREFIX,
            "uuid": SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="probe")),
            "time": SimpleNamespace(
                time=self.clock.time, strftime=time.strftime, localtime=time.localtime
            ),
            "Response": Response,
            "HEALTH_CHECK_TIMEOUT": timeout,
            "logger": logging.getLogger(__name__),
        }
        self.probe_rid = f"{HEALTH_PREFIX}_probe"
        return load_function(HTTP_SERVER, "health_generate", namespace)

    @staticmethod
    def request():
        return SimpleNamespace(url=SimpleNamespace(path="/health_generate"))

    async def flush_cancellation(self):
        for _ in range(4):
            await asyncio.sleep(0)

    async def test_busy_success_aborts_only_probe_and_releases_state(self):
        business = object()
        self.manager.rid_to_state["business"] = business

        async def sleep(_):
            await self.manager.dispatched.wait()
            self.clock.now += 1
            self.manager.last_receive_tstamp = self.clock.now
            await asyncio.sleep(0)

        response = await self.make_endpoint(sleep)(self.request())
        await self.flush_cancellation()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.server_status, "Up")
        self.assertEqual([req.rid for req in self.manager.sent], [self.probe_rid])
        self.assertNotIn(self.probe_rid, self.manager.rid_to_state)
        self.assertIs(self.manager.rid_to_state["business"], business)
        # A late scheduler abort echo is harmless and cannot remove business state.
        self.manager._handle_abort_req(AbortReq(self.probe_rid))
        self.assertIs(self.manager.rid_to_state["business"], business)

    async def test_completed_probe_needs_no_abort(self):
        finished = asyncio.Event()

        async def completed_response(obj, request):
            # Scheduler output removes the state before waking the waiter.
            self.manager.rid_to_state.pop(obj.rid)
            finished.set()
            yield {}

        self.manager._wait_one_response = completed_response

        async def sleep(_):
            await finished.wait()
            self.clock.now += 1
            self.manager.last_receive_tstamp = self.clock.now
            await asyncio.sleep(0)

        response = await self.make_endpoint(sleep)(self.request())
        await self.flush_cancellation()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.sent, [])
        self.assertNotIn(self.probe_rid, self.manager.rid_to_state)

    async def test_timeout_aborts_dispatched_probe(self):
        async def sleep(_):
            await self.manager.dispatched.wait()
            self.clock.now += 1
            await asyncio.sleep(0)

        response = await self.make_endpoint(sleep, timeout=1)(self.request())
        await self.flush_cancellation()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.manager.server_status, "UnHealthy")
        self.assertEqual([req.rid for req in self.manager.sent], [self.probe_rid])
        self.assertNotIn(self.probe_rid, self.manager.rid_to_state)

    async def test_external_cancellation_aborts_dispatched_probe(self):
        async def sleep(_):
            await asyncio.Event().wait()

        endpoint = asyncio.create_task(self.make_endpoint(sleep)(self.request()))
        await self.manager.dispatched.wait()
        endpoint.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await endpoint
        await self.flush_cancellation()
        self.assertEqual([req.rid for req in self.manager.sent], [self.probe_rid])
        self.assertNotIn(self.probe_rid, self.manager.rid_to_state)
        self.assertEqual(self.manager.server_status, "Up")

    async def test_cancel_before_dispatch_drops_state_without_abort(self):
        self.manager.allow_dispatch.clear()

        async def sleep(_):
            await asyncio.Event().wait()

        endpoint = asyncio.create_task(self.make_endpoint(sleep)(self.request()))
        await self.manager.send_entered.wait()
        endpoint.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await endpoint
        await self.flush_cancellation()
        self.assertEqual(self.manager.sent, [])
        self.assertNotIn(self.probe_rid, self.manager.rid_to_state)

    def test_scheduler_busy_shortcut_excludes_abort(self):
        rid = f"{HEALTH_PREFIX}_probe"
        classify = load_function(
            MANAGER_UTILS,
            "is_health_check_generate_req",
            {
                "io_struct": SimpleNamespace(AbortReq=AbortReq),
                "HEALTH_CHECK_RID_PREFIX": HEALTH_PREFIX,
            },
        )
        self.assertTrue(classify(GenerateReqInput(rid)))
        self.assertFalse(classify(AbortReq(rid)))
        self.assertFalse(classify(GenerateReqInput("business")))

        process = load_function(
            SCHEDULER,
            "process_input_requests",
            {
                "time": time,
                "get_mm": lambda: SimpleNamespace(mm_feature_transport="none"),
                "is_health_check_generate_req": classify,
            },
            "Scheduler",
        )
        delivered = []
        scheduler = SimpleNamespace(
            session_controller=SimpleNamespace(maybe_reap=lambda now: None),
            is_fully_idle=lambda for_health_check: False,
            return_health_check_ipcs=[],
            _request_dispatcher=lambda req: delivered.append(req),
            flush_wrapper=SimpleNamespace(check_pending=lambda: None),
            external_corpus_manager=None,
        )
        process(scheduler, [GenerateReqInput(rid), AbortReq(rid)])
        self.assertEqual(len(scheduler.return_health_check_ipcs), 1)
        self.assertEqual(len(delivered), 1)
        self.assertIsInstance(delivered[0], AbortReq)


if __name__ == "__main__":
    unittest.main()
