"""Small CPU interpreter for the production integer-routing kernel bodies.

No Triton import, compiler, accelerator API or SGLang initialization. This checks
integer/mask/store semantics; it cannot establish device lowering or lifetimes.
"""

import ast
import copy

import torch


class _Pointer:
    def __init__(self, data, index=0, unique=False):
        self.data = data.reshape(-1)
        self.index = index
        self.unique = unique

    def __add__(self, index):
        return _Pointer(self.data, self.index + index, self.unique)


class _CPU_TL:
    int32 = torch.int32
    int64 = torch.int64

    def __init__(self):
        self.program = (0, 0)

    def program_id(self, axis):
        return self.program[axis]

    @staticmethod
    def arange(start, stop):
        return torch.arange(start, stop, dtype=torch.int32, device="cpu")

    @staticmethod
    def where(mask, yes, no):
        return torch.where(mask, yes, no)

    @staticmethod
    def sum(value, axis):
        return torch.sum(value, dim=axis, dtype=value.dtype)

    @staticmethod
    def cumsum(value, axis):
        return torch.cumsum(value, dim=axis, dtype=value.dtype)

    @staticmethod
    def _indices(pointer, mask):
        index = torch.as_tensor(pointer.index, dtype=torch.int64, device="cpu")
        mask = torch.broadcast_to(
            torch.as_tensor(True if mask is None else mask, dtype=torch.bool), index.shape
        )
        selected = index[mask]
        if not bool(((selected >= 0) & (selected < pointer.data.numel())).all()):
            raise AssertionError("unmasked out-of-bounds kernel access")
        return index, mask, selected

    @staticmethod
    def load(pointer, mask=None, other=0):
        index, mask, selected = _CPU_TL._indices(pointer, mask)
        out = torch.full(index.shape, other, dtype=pointer.data.dtype, device="cpu")
        out[mask] = pointer.data[selected]
        return out

    @staticmethod
    def store(pointer, value, mask=None):
        index, mask, selected = _CPU_TL._indices(pointer, mask)
        if pointer.unique:
            if selected.numel() != torch.unique(selected).numel():
                raise AssertionError("duplicate destination within kernel program")
            if not bool((pointer.data[selected] == -1).all()):
                raise AssertionError("duplicate destination across kernel programs")
        value = torch.broadcast_to(torch.as_tensor(value), index.shape)
        pointer.data[selected] = value[mask].to(pointer.data.dtype)


class RoutingKernelsCPU:
    """Run the actual four kernel ASTs with CPU tensors and checked pointers."""

    def __init__(self, source_path):
        names = {"_count", "_prefix", "_compact", "_gather"}
        methods = [
            copy.deepcopy(node)
            for node in ast.parse(source_path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if {node.name for node in methods} != names:
            raise AssertionError("routing kernel contract changed")
        for node in methods:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
        self.tl = _CPU_TL()
        namespace = {"tl": self.tl, "__builtins__": {}}
        module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
        exec(compile(module, str(source_path), "exec"), namespace)
        self.kernels = {name: namespace[name] for name in names}

    def plan(self, ids, geometry, rank):
        """Count/prefix/compact, including empty experts and padded block tails."""
        g = geometry
        partial = torch.full((18, g.blocks), -1, dtype=torch.int32)
        offsets = torch.full_like(partial, -1)
        counts = torch.full((18,), -1, dtype=torch.int64)
        total = torch.full((1,), -1, dtype=torch.int32)
        selected = torch.full((g.capacity,), -1, dtype=torch.int32)
        inverse = torch.full((g.world_rows, 8), -1, dtype=torch.int32)
        args = (g.world_rows * 8, g.local_rows * 64, rank * 18, g.active_mask)
        for expert in range(18):
            for block in range(g.blocks):
                self.tl.program = (expert, block)
                self.kernels["_count"](
                    _Pointer(ids), _Pointer(partial, unique=True), *args, g.blocks, 1024
                )
        self.tl.program = (0, 0)
        self.kernels["_prefix"](
            _Pointer(partial), _Pointer(offsets, unique=True),
            _Pointer(counts, unique=True), _Pointer(total, unique=True),
            g.blocks, 1 << (g.blocks - 1).bit_length(), 32,
        )
        for expert in range(18):
            for block in range(g.blocks):
                self.tl.program = (expert, block)
                self.kernels["_compact"](
                    _Pointer(ids), _Pointer(offsets), _Pointer(selected, unique=True),
                    _Pointer(inverse, unique=True), *args, g.blocks, 1024,
                )
        n = int(total[0])  # CPU oracle only; the production path never reads this.
        if not 0 <= n <= g.capacity or n != int(counts.sum()):
            raise AssertionError("invalid count sum/capacity")
        if not bool((selected[:n] >= 0).all()) or not bool((selected[n:] == -1).all()):
            raise AssertionError("compact prefix hole or tail write")
        return selected, counts, inverse, total

    def gather(self, q, scales, gates, selected, total):
        """Interpret INT32 pointer moves, including exceptional FP32 bit patterns."""
        capacity = selected.numel()
        recv = torch.full((capacity, 1024), 123, dtype=torch.int32)
        rs = torch.full((capacity,), 123, dtype=torch.int32)
        weights = torch.full_like(rs, 123)
        rows = torch.full((capacity,), -1, dtype=torch.int64)
        for index in range(capacity):
            self.tl.program = (index, 0)
            self.kernels["_gather"](
                _Pointer(q.view(torch.int32)), _Pointer(scales.view(torch.int32)),
                _Pointer(gates.view(torch.int32)), _Pointer(selected), _Pointer(total),
                _Pointer(recv), _Pointer(rs), _Pointer(rows), _Pointer(weights), 1024,
            )
        return recv.view(torch.int8), rs.view(torch.float32), rows, weights.view(torch.float32)
