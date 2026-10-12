#!/usr/bin/env python3
"""Copy the pinned external kernel and add explicit H4/H8/native-state contracts.

The recurrence, vector arithmetic and double buffering remain the original kernel.
Only a new output tree is written. The versioned Torch op prevents an H8 tensor
from reaching the original hard-coded-H4 binary.
"""
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

PINNED_COMMIT = "4c4cd9f04fab99aa8c75f2590944f71856c23675"
PINNED_SOURCE_SHA256 = (
    "7678d3756baad14350490f4e24f739e233d031486e39b760a8d3778dee75805a"
)


def replace_once(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise ValueError(
            f"source contract changed: expected {count}, found {actual}: {before!r}"
        )
    return source.replace(before, after)


def adapt(source):
    source = replace_once(
        source,
        "        GM_ADDR TRACK_LENS\n    );",
        "        GM_ADDR TRACK_LENS, uint32_t heads, uint32_t state_transposed\n    );",
    )
    source = replace_once(source, "        static constexpr uint32_t H  = 4;\n", "")
    source = replace_once(
        source,
        "    AscendC::GlobalTensor<int32_t> indicesGm;",
        "    uint32_t heads_;\n    bool state_transposed_;\n    AscendC::GlobalTensor<int32_t> indicesGm;",
    )
    source = replace_once(
        source,
        "    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS) \n{",
        "    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS,\n    uint32_t heads, uint32_t state_transposed) \n{",
    )
    source = replace_once(
        source,
        "    qnGm.SetGlobalBuffer(",
        "    heads_ = heads;\n    state_transposed_ = state_transposed != 0;\n    qnGm.SetGlobalBuffer(",
    )
    source = replace_once(
        source,
        "    constexpr uint32_t H = config::H;",
        "    const uint32_t H = heads_;",
    )
    source = replace_once(
        source,
        "        + static_cast<uint64_t>(iv*BV);",
        "        + static_cast<uint64_t>(iv*BV) * (state_transposed_ ? D : 1);",
    )
    source = replace_once(
        source,
        "                + static_cast<uint64_t>(iv) * BV;",
        "                + static_cast<uint64_t>(iv) * BV * (state_transposed_ ? D : 1);",
    )
    source = replace_once(
        source,
        "    // Step 1: GM [128,128] -> UB [128,24]",
        "    if (state_transposed_) {\n"
        "        AscendC::DataCopy(state, stateGm[gmOffset], config::BV * STATE_STRIDE);\n"
        "        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);\n"
        "        return;\n    }\n    // Step 1: GM [128,128] -> UB [128,24]",
    )
    source = replace_once(
        source,
        "    // Step 1: UB [16,144] -> UB [128,24]",
        "    if (state_transposed_) {\n"
        "        AscendC::PipeBarrier<PIPE_V>();\n"
        "        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);\n"
        "        AscendC::DataCopy(stateGm[gmOffset], state, BV * STATE_STRIDE);\n"
        "        return;\n    }\n    // Step 1: UB [16,144] -> UB [128,24]",
    )
    source = replace_once(
        source,
        "    GM_ADDR TRACK_LENS              // int32/int64\n)",
        "    GM_ADDR TRACK_LENS, uint32_t heads, uint32_t state_transposed\n)",
        count=2,
    )
    source = replace_once(
        source,
        "            TRACK_LENS\n    );",
        "            TRACK_LENS, heads, state_transposed\n    );",
        count=2,
    )
    source = replace_once(
        source, "    constexpr int32_t H = 4;", "    const int32_t H = qn.size(2);"
    )
    source = replace_once(
        source,
        "    constexpr int32_t blockDim = NUM_V_TILES * NUM_REQ * H;",
        "    const int32_t blockDim = NUM_V_TILES * NUM_REQ * H;\n"
        + r"""
    TORCH_CHECK(qn.dim() == 4 && qn.size(0) == 1 && qn.size(3) == D && (H == 4 || H == 8),
                "prepared recurrent v1 requires B1 H4/H8 D128");
    auto checkInput = [&](const at::Tensor &tensor, at::ScalarType dtype, const char *name) {
        TORCH_CHECK(tensor.device() == qn.device() && tensor.scalar_type() == dtype && tensor.is_contiguous(),
                    name, " device/dtype/contiguous contract failed");
    };
    checkInput(qn, at::kFloat, "qn"); checkInput(kn, at::kFloat, "kn");
    checkInput(gdecay, at::kFloat, "gdecay"); checkInput(beta, at::kFloat, "beta");
    checkInput(v, at::kBFloat16, "v"); checkInput(out, at::kBFloat16, "out");
    TORCH_CHECK(kn.sizes() == qn.sizes() && v.sizes() == qn.sizes() && out.sizes() == qn.sizes()
                && gdecay.sizes() == qn.sizes() && beta.dim() == 3 && beta.size(0) == 1
                && beta.size(1) == qn.size(1) && beta.size(2) == H, "input shapes differ");
    TORCH_CHECK(state.device() == qn.device() && state.scalar_type() == at::kFloat
                && state.dim() == 4 && state.size(1) == H && state.size(2) == D && state.size(3) == D
                && state.stride(0) == H*D*D && state.stride(1) == D*D
                && ((state.stride(2) == D && state.stride(3) == 1)
                    || (state.stride(2) == 1 && state.stride(3) == D)), "native FP32 state layout unsupported");
    TORCH_CHECK(indices.device() == qn.device() && indices.is_contiguous() && indices.numel() == 1
                && (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong), "state index contract failed");
    checkInput(starts, at::kInt, "starts");
    TORCH_CHECK(starts.numel() == 2, "B1 starts must have two values");
    auto indices32 = indices.to(at::kInt);
    const uint32_t state_transposed = state.stride(2) == 1;
    at::Tensor trackIndices32, trackLens32;
""",
    )
    source = replace_once(
        source,
        "                tensor.scalar_type() == at::kInt,\n",
        "                tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,\n",
    )
    source = replace_once(
        source,
        "        trackIndicesPtr = gm(*track_indices);\n        trackLensPtr = gm(*track_lens);",
        "        trackIndices32 = track_indices->to(at::kInt);\n"
        "        trackLens32 = track_lens->to(at::kInt);\n"
        "        trackIndicesPtr = gm(trackIndices32);\n        trackLensPtr = gm(trackLens32);",
    )
    source = replace_once(
        source,
        "            gm(indices), gm(starts),",
        "            gm(indices32), gm(starts),",
        count=2,
    )
    source = replace_once(
        source,
        "            trackIndicesPtr, trackLensPtr);",
        "            trackIndicesPtr, trackLensPtr, H, state_transposed);",
    )
    source = replace_once(
        source,
        "            nullptr, nullptr);",
        "            nullptr, nullptr, H, state_transposed);",
    )
    source = replace_once(
        source,
        "TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {",
        "TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {\n"
        '    m.def("prepared_recurrent_v1(Tensor qn, Tensor kn, Tensor v, "\n'
        '          "Tensor gdecay, Tensor beta, Tensor(a!) out, Tensor(b!) state, "\n'
        '          "Tensor indices, Tensor starts, Tensor? track_indices=None, "\n'
        '          "Tensor? track_lens=None, bool track_state=False) -> ()");',
    )
    source = replace_once(
        source,
        '    m.impl("prepared_recurrent", &prepared_recurrent);',
        '    m.impl("prepared_recurrent", &prepared_recurrent);\n'
        '    m.impl("prepared_recurrent_v1", &prepared_recurrent);',
    )
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original", type=Path, required=True, help="sgl-glm53-ascend-ops root"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    source_path = args.original / "csrc/kda/prepared_recurrent.cpp"
    if not source_path.is_file():
        parser.error(f"missing original source: {source_path}")
    commit = subprocess.run(
        ["git", "-C", str(args.original), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        parser.error(f"expected external commit {PINNED_COMMIT}, found {commit}")
    original_bytes = source_path.read_bytes()
    digest = hashlib.sha256(original_bytes).hexdigest()
    if digest != PINNED_SOURCE_SHA256:
        parser.error(
            f"expected original source sha256 {PINNED_SOURCE_SHA256}, found {digest}"
        )
    original = original_bytes.decode("utf-8")
    updated = adapt(original)
    args.output.mkdir(parents=True)
    shutil.copy2(args.original / "CMakeLists.txt", args.output / "CMakeLists.txt")
    shutil.copytree(args.original / "csrc", args.output / "csrc")
    shutil.copytree(
        args.original / "src",
        args.output / "src",
        ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"),
    )
    (args.output / "csrc/kda/prepared_recurrent.cpp").write_text(updated)
    print("external_commit", commit)
    print("original_sha256", digest)
    print("compatible_sha256", hashlib.sha256(updated.encode()).hexdigest())
