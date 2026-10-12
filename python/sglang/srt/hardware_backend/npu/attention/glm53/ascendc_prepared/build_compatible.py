#!/usr/bin/env python3
"""Build an isolated, versioned adapter for the pinned external AscendC kernel.

Keep the upstream H4 launch grid and compile-time HEAD_OFFSET/H_TOTAL variants.
For H8, v2 makes the two upstream H4 launches on the same NPU stream.
"""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

PINNED_COMMIT = "6fedd85fdf61481437e64b1b635371d9c8201b14"
PINNED_SOURCE_SHA256 = (
    "da0c2e1513debf025bc122c47eefba3db547853b905249424c7fbbcecbacbe71"
)


def replace_once(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise ValueError(
            f"source contract changed: expected {count}, found {actual}: {before!r}"
        )
    return source.replace(before, after)


HOST = r"""
// Both public ops use exactly one H4 grid per launch (8 value tiles x 4 heads).
// v2 submits two ordered launches for H8, preserving the upstream grouping.
static void prepared_recurrent_impl(
    const at::Tensor &qn,
    const at::Tensor &kn,
    const at::Tensor &v,
    const at::Tensor &gdecay,
    const at::Tensor &beta,
    at::Tensor &out,
    at::Tensor &state,
    const at::Tensor &indices,
    const at::Tensor &starts,
    const c10::optional<at::Tensor> &track_indices,
    const c10::optional<at::Tensor> &track_lens,
    bool track_state,
    int64_t head_offset,
    bool full_h8)
{
    constexpr int32_t D = 128;
    constexpr int32_t BV = 16;
    constexpr int32_t HEADS_PER_LAUNCH = 4;
    constexpr int32_t blockDim = (D / BV) * HEADS_PER_LAUNCH;

    TORCH_CHECK(qn.is_privateuseone(), "qn must be an NPU tensor");
    TORCH_CHECK(qn.dim() == 4 && qn.size(0) == 1 && qn.size(3) == D
                && (qn.size(2) == 4 || qn.size(2) == 8)
                && qn.size(1) >= 128 && qn.size(1) <= 16384,
                "prepared recurrent requires B1, T128..16384, H4/H8, D128");
    const int64_t heads = qn.size(2);
    TORCH_CHECK(state.dim() == 4 && state.size(0) >= 1
                && state.size(1) == heads && state.size(2) == D
                && state.size(3) == D, "state must be [slots,H,128,128]");
    TORCH_CHECK(full_h8 ? (head_offset == 0)
                        : (head_offset == 0 || (heads == 8 && head_offset == 4)),
                "invalid head_offset for prepared recurrent");

    auto checkInput = [&](const at::Tensor &tensor, at::ScalarType dtype,
                          const char *name) {
        TORCH_CHECK(tensor.device() == qn.device()
                    && tensor.scalar_type() == dtype && tensor.is_contiguous(),
                    name, " device/dtype/contiguous contract failed");
    };
    checkInput(qn, at::kFloat, "qn");
    checkInput(kn, at::kFloat, "kn");
    checkInput(gdecay, at::kFloat, "gdecay");
    checkInput(beta, at::kFloat, "beta");
    checkInput(v, at::kBFloat16, "v");
    checkInput(out, at::kBFloat16, "out");
    TORCH_CHECK(kn.sizes() == qn.sizes()
                && gdecay.sizes() == qn.sizes()
                && v.sizes() == qn.sizes()
                && out.sizes() == qn.sizes()
                && beta.dim() == 3 && beta.size(0) == 1
                && beta.size(1) == qn.size(1) && beta.size(2) == heads,
                "prepared recurrent input shapes differ");
    TORCH_CHECK(state.device() == qn.device()
                && state.scalar_type() == at::kFloat
                && state.stride(0) == heads * D * D
                && state.stride(1) == D * D
                && ((state.stride(2) == D && state.stride(3) == 1)
                    || (state.stride(2) == 1 && state.stride(3) == D)),
                "native FP32 state layout unsupported");
    const uint32_t state_transposed = state.stride(2) == 1;

    auto checkMetadata = [&](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.device() == qn.device()
                    && tensor.is_contiguous() && tensor.dim() == 1
                    && tensor.numel() == 1
                    && (tensor.scalar_type() == at::kInt
                        || tensor.scalar_type() == at::kLong),
                    name, " must be contiguous int32/int64 [1] on qn device");
    };
    checkMetadata(indices, "indices");
    TORCH_CHECK(starts.device() == qn.device()
                && starts.scalar_type() == at::kInt
                && starts.is_contiguous() && starts.dim() == 1
                && starts.numel() == 2,
                "starts must be contiguous int32 [2] on qn device");
    if (track_state) {
        TORCH_CHECK(track_indices.has_value() && track_indices->defined()
                    && track_lens.has_value() && track_lens->defined(),
                    "tracking requires track_indices and track_lens");
        checkMetadata(*track_indices, "track_indices");
        checkMetadata(*track_lens, "track_lens");
    }

    // Keep converted tensors alive until both launches have been enqueued.
    at::Tensor indices32 = indices.to(at::kInt);
    at::Tensor trackIndices32, trackLens32;
    auto gm = [](const at::Tensor &tensor) -> uint8_t* {
        return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.data_ptr()));
    };
    uint8_t *trackIndicesPtr = nullptr;
    uint8_t *trackLensPtr = nullptr;
    if (track_state) {
        trackIndices32 = track_indices->to(at::kInt);
        trackLens32 = track_lens->to(at::kInt);
        trackIndicesPtr = gm(trackIndices32);
        trackLensPtr = gm(trackLens32);
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

#define LAUNCH(KERNEL) \
    KERNEL<<<blockDim, nullptr, stream>>>( \
        gm(qn), gm(kn), gm(v), gm(gdecay), gm(beta), \
        gm(out), gm(state), gm(indices32), gm(starts), \
        trackIndicesPtr, trackLensPtr, state_transposed)

    if (heads == 4) {
        if (track_state) {
            LAUNCH(prepared_recurrent_kernel_track_h4_o0);
        } else {
            LAUNCH(prepared_recurrent_kernel_h4_o0);
        }
    } else {
        if (full_h8 || head_offset == 0) {
            if (track_state) {
                LAUNCH(prepared_recurrent_kernel_track_h8_o0);
            } else {
                LAUNCH(prepared_recurrent_kernel_h8_o0);
            }
        }
        if (full_h8 || head_offset == 4) {
            if (track_state) {
                LAUNCH(prepared_recurrent_kernel_track_h8_o4);
            } else {
                LAUNCH(prepared_recurrent_kernel_h8_o4);
            }
        }
    }
#undef LAUNCH
}

void prepared_recurrent(
    const at::Tensor &qn, const at::Tensor &kn, const at::Tensor &v,
    const at::Tensor &gdecay, const at::Tensor &beta, at::Tensor &out,
    at::Tensor &state, const at::Tensor &indices, const at::Tensor &starts,
    const c10::optional<at::Tensor> &track_indices,
    const c10::optional<at::Tensor> &track_lens,
    bool track_state, int64_t head_offset)
{
    prepared_recurrent_impl(qn, kn, v, gdecay, beta, out, state, indices,
                            starts, track_indices, track_lens, track_state,
                            head_offset, false);
}

void prepared_recurrent_v2(
    const at::Tensor &qn, const at::Tensor &kn, const at::Tensor &v,
    const at::Tensor &gdecay, const at::Tensor &beta, at::Tensor &out,
    at::Tensor &state, const at::Tensor &indices, const at::Tensor &starts,
    const c10::optional<at::Tensor> &track_indices,
    const c10::optional<at::Tensor> &track_lens, bool track_state)
{
    prepared_recurrent_impl(qn, kn, v, gdecay, beta, out, state, indices,
                            starts, track_indices, track_lens, track_state,
                            0, true);
}

"""


def adapt(source):
    # For native transposed state [slot, head, value, key], a value tile is
    # already contiguous. The original staging transpose remains for [K,V].
    source = replace_once(
        source,
        "    AscendC::GlobalTensor<int32_t> indicesGm;",
        "    bool state_transposed_;\n    AscendC::GlobalTensor<int32_t> indicesGm;",
    )
    source = replace_once(
        source,
        "        GM_ADDR TRACK_LENS\n    );",
        "        GM_ADDR TRACK_LENS, uint32_t state_transposed\n    );",
    )
    source = replace_once(
        source,
        "    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS)\n{",
        "    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS,\n"
        "    uint32_t state_transposed)\n{\n"
        "    state_transposed_ = state_transposed != 0;",
    )
    source = replace_once(
        source,
        "    // Step 1: GM [128,128] -> UB [128,24]",
        "    if (state_transposed_) {\n"
        "        AscendC::DataCopy(state, stateGm[gmOffset], config::BV * STATE_STRIDE);\n"
        "        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);\n"
        "        return;\n    }\n"
        "    // Step 1: GM [128,128] -> UB [128,24]",
    )
    source = replace_once(
        source,
        "    // Step 1: UB [16,144] -> UB [128,24]",
        "    if (state_transposed_) {\n"
        "        AscendC::PipeBarrier<PIPE_V>();\n"
        "        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);\n"
        "        AscendC::DataCopy(stateGm[gmOffset], state, BV * STATE_STRIDE);\n"
        "        return;\n    }\n"
        "    // Step 1: UB [16,144] -> UB [128,24]",
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
        "    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS                          \\",
        "    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS,                         \\\n"
        "    uint32_t state_transposed                                         \\",
    )
    source = replace_once(
        source,
        "            TRACK_INDICES, TRACK_LENS);",
        "            TRACK_INDICES, TRACK_LENS, state_transposed);",
    )
    start = source.index("void prepared_recurrent(\n")
    end = source.index("TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend, m) {", start)
    source = source[:start] + HOST + source[end:]
    source = replace_once(
        source,
        '          "bool track_state=False, int head_offset=0) -> ()");',
        '          "bool track_state=False, int head_offset=0) -> ()");\n'
        '    m.def("prepared_recurrent_v2(Tensor qn, Tensor kn, Tensor v, "\n'
        '          "Tensor gdecay, Tensor beta, Tensor(a!) out, Tensor(b!) state, "\n'
        '          "Tensor indices, Tensor starts, "\n'
        '          "Tensor? track_indices=None, Tensor? track_lens=None, "\n'
        '          "bool track_state=False) -> ()");',
    )
    source = replace_once(
        source,
        '    m.impl("prepared_recurrent", &prepared_recurrent);',
        '    m.impl("prepared_recurrent", &prepared_recurrent);\n'
        '    m.impl("prepared_recurrent_v2", &prepared_recurrent_v2);',
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
    updated = adapt(original_bytes.decode("utf-8"))
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
