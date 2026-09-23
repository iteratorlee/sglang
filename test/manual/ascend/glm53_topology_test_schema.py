"""Source-derived config shapes for CPU tests without msgspec/torch installed.

This is not a full ServerArgs construction or runtime publish test. Only input
annotations from the real _INPUT_NAMESPACES enter ServerArgsShape; unannotated
Derived declarations cannot be supplied as fake input fields. ParallelContext,
its property installer and width arithmetic are executed from production AST.
"""

import ast
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
SERVER_TREE = ast.parse((SRT / "server_args.py").read_text())
INPUT_NAMESPACES = {
    item.id
    for node in SERVER_TREE.body
    if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "_INPUT_NAMESPACES" for t in node.targets)
    for item in node.value.elts
}
DECLARATIONS = {
    node.name: node
    for path in (SRT / "arg_groups/fields").glob("*.py")
    for node in ast.parse(path.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name in INPUT_NAMESPACES
}
assert DECLARATIONS.keys() == INPUT_NAMESPACES


def input_fields(declaration):
    return {
        node.target.id
        for node in declaration.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


SERVER_FIELDS = frozenset().union(
    *(input_fields(node) for node in DECLARATIONS.values()),
    *(
        input_fields(node)
        for node in SERVER_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    ),
)
ServerArgsShape = type(
    "ServerArgsShape", (), {"__slots__": tuple(sorted(SERVER_FIELDS))}
)
PARALLEL_INPUT_FIELDS = input_fields(DECLARATIONS["Parallel"])
DERIVED_NODES = [
    node
    for node in DECLARATIONS["Parallel"].body
    if isinstance(node, ast.Assign)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Name)
    and node.value.func.id == "Derived"
]
PARALLEL_DERIVED_FIELDS = {node.targets[0].id for node in DERIVED_NODES}


def server_args_shape(**values):
    cfg = ServerArgsShape()
    for name, value in values.items():
        setattr(cfg, name, value)  # Slots reject any nonexistent input field.
    return cfg


def runtime_namespace():
    tree = ast.parse((SRT / "runtime_context.py").read_text())
    names = {
        "derive_attention_widths",
        "derive_parallel_widths",
        "parallel_widths_of",
        "attn_tp_size_of",
        "ParallelContext",
        "_install_derived_widths",
        "get_parallel",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    installer = next(node for node in nodes if node.name == "_install_derived_widths")
    # Supply only metadata imports locally; execute the real installer body.
    installer.body = [
        node for node in installer.body if not isinstance(node, ast.ImportFrom)
    ]
    metadata = {"Derived": namedtuple("Derived", "fn doc")}
    exec(
        compile(
            ast.Module(body=DERIVED_NODES, type_ignores=[]),
            "Parallel.declarations",
            "exec",
        ),
        metadata,
    )
    ns = {
        "contextmanager": contextmanager,
        "Derived": metadata["Derived"],
        "Parallel": type(
            "ParallelDeclarations",
            (),
            {name: metadata[name] for name in PARALLEL_DERIVED_FIELDS},
        ),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(SRT / "runtime_context.py"), "exec"), ns)
    ns["_install_derived_widths"]()
    return ns


RUNTIME = runtime_namespace()


def published_parallel(cfg):
    """Project declared inputs and real derived widths into a test config bag."""
    values = {
        name: getattr(cfg, name)
        for name in PARALLEL_INPUT_FIELDS
        if hasattr(cfg, name)
    }
    widths = RUNTIME["parallel_widths_of"](cfg)
    assert widths.keys() == PARALLEL_DERIVED_FIELDS
    values.update(widths)
    bag = type(
        "ParallelConfigShape",
        (),
        {"__slots__": tuple(values), "_fields": frozenset(values)},
    )()
    for name, value in values.items():
        setattr(bag, name, value)
    parallel = RUNTIME["ParallelContext"]()
    parallel._config = bag
    return parallel
