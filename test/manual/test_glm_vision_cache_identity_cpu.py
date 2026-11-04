"""CPU cache-identity contracts, using production code without device imports.

AST loading excludes accelerator/model initialization imports, not the tested
method bodies. Hashing, msgspec items, splitting, finalization, cache lookup and
chunk assembly are the real implementations. The small encoder below is only
a grid-sensitive sentinel; it is NOT a GLM numerical or accuracy oracle.
"""

# Imports also supply globals to verbatim production AST definitions.
# ruff: noqa: F401

import abc
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import unittest
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "python/sglang/srt"


def source_nodes(relative, namespace, names=(), methods=None):
    """Execute selected production definitions verbatim, with lazy annotations."""
    path = SRT / relative
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if getattr(node, "name", None) in names:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (methods or {}):
            node.body = [
                child
                for child in node.body
                if getattr(child, "name", None) in methods[node.name]
            ]
            if not node.body:
                node.body = [ast.copy_location(ast.Pass(), node)]
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(future + selected, []), str(path), "exec"), namespace)


def production_code():
    ns = dict(globals())
    ns["logger"] = logging.getLogger(__name__)
    ns["nn"] = torch.nn
    source_nodes("utils/common.py", ns, ["flatten_nested_list"])
    source_nodes(
        "managers/schedule_batch.py",
        ns,
        [
            "MM_PAD_SHIFT_VALUE",
            "_MM_HASH_MASK",
            "_compute_pad_value",
            "Modality",
            "MultimodalInputFormat",
            "MultimodalDataItem",
        ],
    )
    source_nodes(
        "managers/mm_utils.py",
        ns,
        [
            "data_hash",
            "tensor_hash",
            "hash_feature",
            "_get_length",
            "_is_rank2_grid",
            "_slice_value",
            "_slice_model_data",
            "_compute_patch_slices",
            "_PATCH_ALIGNED_KEYS",
            "_split_model_data_for_item",
            "_try_simple_split",
            "get_new_expanded_mm_items",
        ],
    )
    # These types are referenced only by isinstance checks. No transport is
    # constructed in these CPU tests.
    ns["ShmPointerMMData"] = type("UnusedShmPointer", (), {})
    ns["CudaIpcTensorTransportProxy"] = type("UnusedCudaProxy", (), {})
    source_nodes(
        "multimodal/cache/identity.py",
        ns,
        [
            "CONTENT_HASH_PREFIX",
            "_SHA256_HEX_LENGTH",
            "parse_content_hash",
            "_digest_bytes",
            "_hash_parts",
            "resolve_multimodal_item_hash",
        ],
    )
    source_nodes(
        "mem_cache/multimodal_cache.py",
        ns,
        [
            "MultimodalCache",
            "_get_tensor_size",
            "EmbeddingResult",
            "MultiModalStaticCache",
        ],
    )
    source_nodes(
        "multimodal/processors/base_processor.py",
        ns,
        ["BaseMultiModalProcessorOutput", "MultimodalSpecialTokens"],
        methods={
            "BaseMultimodalProcessor": {
                "_resolve_processor",
                "_get_preprocessed_input_format",
                "_ensure_input_ids_is_tensor",
                "_process_and_collect_mm_items",
                "collect_mm_items_from_processor_output",
                "process_and_combine_mm_data",
                "get_mm_items_offset",
                "get_mm_items_offset_by_pair",
                "get_mm_item_offsets",
                "_finalize_mm_items",
                "_postprocess_mm_items_before_transport",
                "_precompute_hashes_before_cpu_transfer",
                "_move_feature_to_cpu",
                "_prepare_mm_items_for_transport",
            }
        },
    )
    ns["SGLangBaseProcessor"] = ns["BaseMultimodalProcessor"]
    # Discover local helpers so the identical test also runs against the base
    # revision (which has no GLM identity override).
    glm_tree = ast.parse((SRT / "multimodal/processors/glm4v.py").read_text())
    helpers = [
        node.name
        for node in glm_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_glm_vision_cache")
    ]
    source_nodes(
        "multimodal/processors/glm4v.py",
        ns,
        helpers,
        methods={
            "Glm4vImageProcessor": {
                "_postprocess_mm_items_before_transport",
                "get_mm_item_offsets",
            }
        },
    )
    source_nodes(
        "models/glm4v.py",
        ns,
        methods={"Glm4vVisionModel": {"rot_pos_emb"}},
    )
    source_nodes(
        "models/glm5_next.py",
        ns,
        methods={
            "Glm5NextForConditionalGeneration": {
                "get_image_feature",
                "get_video_feature",
            }
        },
    )
    source_nodes(
        "managers/mm_schedule.py",
        ns,
        [
            "get_embedding_chunk",
            "_embedding_token_count",
            "_discard_mismatched_cached_embedding",
            "_can_skip_pre_embed_feature_move",
            "_move_items_to_device",
            "_acknowledge_deferred_cuda_ipc_cache_hits",
            "_get_chunked_embedding_by_item",
            "_get_chunked_embedding_full",
            "_flatten_embedding_result",
            "_get_precomputed_embedding",
        ],
    )
    # No TP transport acknowledgement is needed on this non-owner CPU rank.
    ns["get_parallel"] = lambda: SimpleNamespace(attn_tp_rank=1)
    ns["EVSEmbeddingResult"] = type("UnusedEVSResult", (), {})
    env_spec = importlib.util.spec_from_file_location("glm_cpu_env", SRT / "environ.py")
    env = importlib.util.module_from_spec(env_spec)
    env_spec.loader.exec_module(env)
    ns["envs"] = env.envs
    modules = {"sglang.srt.environ": env}
    for name, exports in {
        "sglang.srt.multimodal.cache": ["resolve_multimodal_item_hash"],
        "sglang.srt.managers.mm_utils": ["hash_feature", "get_new_expanded_mm_items"],
    }.items():
        module = ModuleType(name)
        module.__dict__.update({key: ns[key] for key in exports})
        modules[name] = module
    return ns, modules


class GlmVisionCacheIdentityCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.code, cls.modules = production_code()

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, self.modules))
        self.enterContext(patch.dict(os.environ, {"SGLANG_MM_SKIP_COMPUTE_HASH": "0"}))
        self.ns = self.code
        self.processor = self.ns["Glm4vImageProcessor"]()
        self.processor.precompute_hash_before_cpu_transfer = False
        self.processor.keep_mm_features_on_device = False
        self.processor.use_cuda_ipc = False
        self.processor._processor = None
        self.processor._tokenizer = None
        self.processor.IM_TOKEN_ID = 99
        self.processor.VIDEO_START_TOKEN_ID = 101
        self.processor.VIDEO_END_TOKEN_ID = 102
        self.processor.preserve_processor_input_ids = False
        self.processor.FEATURE_NAMES = ["pixel_values", "pixel_values_videos"]
        self.processor.ATTR_NAME_TO_MODALITY = {
            "pixel_values": self.ns["Modality"].IMAGE,
            "image_grid_thw": self.ns["Modality"].IMAGE,
            "pixel_values_videos": self.ns["Modality"].VIDEO,
            "video_grid_thw": self.ns["Modality"].VIDEO,
        }
        self.ns["embedding_cache"] = self.ns["MultiModalStaticCache"](1024 * 1024)

    def item(self, grids=((1, 2, 8),), *, video=False, offsets=None, feature=None):
        grid = torch.tensor(grids, dtype=torch.int64)
        patches = int(grid.prod(dim=-1).sum())
        feature = (
            torch.arange(patches * 3, dtype=torch.float32).reshape(patches, 3)
            if feature is None
            else feature
        )
        return self.ns["MultimodalDataItem"](
            modality=self.ns["Modality"].VIDEO if video else self.ns["Modality"].IMAGE,
            feature=feature,
            offsets=offsets or [(2, 2 + patches // 4 - 1)],
            model_specific_data={"video_grid_thw" if video else "image_grid_thw": grid},
        )

    def finalize(self, items, *, split=False):
        if split:
            items = self.ns["get_new_expanded_mm_items"](items)
        items = self.processor._finalize_mm_items(items, images=None)
        for item in items:
            item.set_pad_value()
        return items

    def encode(self, items, calls):
        """Grid-sensitive CPU sentinel, deliberately not a vision model."""
        calls.append([item.hash for item in items])
        results = []
        for item in items:
            grid = item.model_specific_data.get("image_grid_thw")
            if grid is None:
                grid = item.video_grid_thw
            count = sum(end - start + 1 for start, end in item.offsets)
            results.append(item.feature[:count] + float(grid[0, 1]) * 100)
        return torch.cat(results)

    def cached(self, items, calls, *, full=False, prefix=0, length=100):
        fn = self.ns[
            "_get_chunked_embedding_full" if full else "_get_chunked_embedding_by_item"
        ]
        kwargs = dict(
            data_embedding_func=lambda batch: self.encode(batch, calls),
            embedding_items_per_req=items,
            items_offset=[offset for item in items for offset in item.offsets],
            extend_prefix_len=prefix,
            extend_seq_len=length,
            device=torch.device("cpu"),
        )
        if full:
            return fn(**kwargs, input_ids=torch.zeros(100, dtype=torch.long))[0]
        return fn(**kwargs)

    def test_legacy_hash_alias_is_real_for_equal_bytes_and_token_count(self):
        a, b = self.item(), self.item(((1, 4, 4),))
        a.set_pad_value()
        b.set_pad_value()
        self.assertTrue(torch.equal(a.feature, b.feature))
        self.assertEqual(a.offsets, b.offsets)
        self.assertEqual(a.hash, b.hash)
        self.assertEqual(a.pad_value, b.pad_value)
        calls = []
        first = self.cached([a], calls)
        stale = self.cached([b], calls)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(first, stale))
        self.assertFalse(torch.equal(stale, self.encode([b], [])))
        print(
            f"legacy alias: hash={a.hash}, pad={a.pad_value}, tokens=4, encoder_calls=1"
        )

    def test_different_grid_misses_same_grid_hits_and_prompt_shift_reuses(self):
        for video in (False, True):
            with self.subTest(video=video):
                a, b, again = self.finalize(
                    [
                        self.item(video=video),
                        self.item(((1, 4, 4),), video=video),
                        self.item(video=video, offsets=[(40, 43)]),
                    ]
                )
                self.assertNotEqual(a.hash, b.hash)
                self.assertNotEqual(a.pad_value, b.pad_value)
                self.assertEqual(a.hash, again.hash)
                calls = []
                first = self.cached([a], calls)
                second = self.cached([b], calls)
                reused = self.cached([again], calls, prefix=41, length=2)
                self.assertEqual(len(calls), 2)
                self.assertFalse(torch.equal(first, second))
                torch.testing.assert_close(second, self.encode([b], []))
                torch.testing.assert_close(reused, first[1:3])

    def test_early_hash_pad_and_repeated_finalization_are_consistent(self):
        a, b = self.item(), self.item(((1, 4, 4),))
        a.set_pad_value()
        b.set_pad_value()
        legacy = a.hash
        self.processor.precompute_hash_before_cpu_transfer = True
        self.finalize([a, b])
        self.assertNotEqual(a.hash, legacy)
        self.assertNotEqual(a.hash, b.hash)
        first = [(x.hash, x.pad_value) for x in (a, b)]
        self.finalize([a, b])
        self.assertEqual(first, [(x.hash, x.pad_value) for x in (a, b)])
        for item in (a, b):
            self.assertEqual(item.pad_value, self.ns["_compute_pad_value"](item.hash))

    def test_shape_dtype_feature_bytes_and_segments_are_covered(self):
        a = self.item()
        same_bytes = [
            self.item(feature=a.feature.reshape(8, 6)),
            self.item(feature=a.feature.view(torch.int32)),
            self.item(((2, 2, 4),)),
            self.item(((1, 2, 4), (1, 2, 4))),
        ]
        for b in same_bytes:
            self.assertEqual(
                self.ns["hash_feature"](a.feature), self.ns["hash_feature"](b.feature)
            )
        changed = self.item()
        changed.feature[-1, -1] += 1
        items = self.finalize([a, *same_bytes, changed])
        self.assertEqual(len({item.hash for item in items}), len(items))

    def test_feature_segment_layout_and_dp_video_metadata_are_covered(self):
        feature = self.item().feature
        a = self.item(feature=[feature[:8], feature[8:]])
        b = self.item(feature=[feature[:4], feature[4:]])
        self.assertEqual(
            self.ns["hash_feature"](a.feature), self.ns["hash_feature"](b.feature)
        )
        self.finalize([a, b])
        self.assertNotEqual(a.hash, b.hash)
        videos = [self.item(video=True) for _ in range(4)]
        for item, n_units, counts in zip(
            videos[1:], [2, 3, 2], [[1, 1], [1, 2], [2, 0]]
        ):
            item.dp_decode_sharded = True
            item.dp_meta = {"n_units": n_units, "gpu_sample_counts": counts}
        self.finalize(videos)
        self.assertEqual(len({item.hash for item in videos}), 4)

    def test_noncontiguous_and_numpy_features_are_content_stable(self):
        a = self.item()
        storage = torch.empty((16, 6), dtype=a.feature.dtype)
        storage[:, ::2] = a.feature
        b = self.item(feature=storage[:, ::2])
        self.assertFalse(b.feature.is_contiguous())
        a, b = self.finalize([a, b])
        self.assertEqual(a.hash, b.hash)
        x = self.item(feature=a.feature.numpy())
        y = self.item(feature=a.feature.numpy().copy())
        x, y = self.finalize([x, y])
        self.assertEqual(x.hash, y.hash)

    def test_bundled_images_rehash_after_split_even_with_early_pad(self):
        bundled = self.item(((1, 2, 8), (1, 4, 4)), offsets=[(2, 5), (20, 23)])
        bundled.feature[16:] = bundled.feature[:16]
        bundled.set_pad_value()
        split = self.finalize([bundled], split=True)
        independent = self.finalize([self.item(), self.item(((1, 4, 4),))])
        self.assertEqual(len(split), 2)
        self.assertEqual([x.hash for x in split], [x.hash for x in independent])
        self.assertNotEqual(split[0].hash, split[1].hash)
        self.assertEqual([x.offsets for x in split], [[(2, 5)], [(20, 23)]])
        self.assertEqual(
            [x.pad_value for x in split], [x.pad_value for x in independent]
        )

    def test_video_grouping_and_ordered_combined_cache(self):
        bundled = self.item(
            ((2, 2, 8), (2, 4, 4)),
            video=True,
            offsets=[(2, 5), (8, 11), (20, 23), (26, 29)],
        )
        bundled.feature[32:] = bundled.feature[:32]
        bundled.set_pad_value()
        videos = self.finalize([bundled], split=True)
        self.assertEqual(len(videos), 2)
        self.assertNotEqual(videos[0].hash, videos[1].hash)
        self.assertEqual(videos[0].offsets, [(2, 5), (8, 11)])
        calls = []
        first = self.cached(videos, calls, full=True)
        reused = self.cached(videos, calls, full=True)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(first, reused)
        reversed_items = [copy.copy(videos[1]), copy.copy(videos[0])]
        reversed_items[0].offsets = videos[0].offsets
        reversed_items[1].offsets = videos[1].offsets
        reordered = self.cached(reversed_items, calls, full=True)
        self.assertEqual(len(calls), 2)
        self.assertFalse(torch.equal(first, reordered))

    def test_multi_offset_full_cache_rejects_same_length_different_video_grid(self):
        a, b, again = self.finalize(
            [
                self.item(((2, 2, 8),), video=True, offsets=[(2, 5), (8, 11)]),
                self.item(((2, 4, 4),), video=True, offsets=[(2, 5), (8, 11)]),
                self.item(((2, 2, 8),), video=True, offsets=[(12, 15), (18, 21)]),
            ]
        )
        calls = []
        first = self.cached([a], calls, full=True)
        second = self.cached([b], calls, full=True)
        reused = self.cached([again], calls, full=True)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(first, reused)
        self.assertFalse(torch.equal(first, second))

    def test_real_processor_collection_splitting_and_mixed_offsets(self):
        image = self.item()
        # Two images and a two-frame video use the same placeholder token.
        ids = (
            [0]
            + [99] * 4
            + [7]
            + [99] * 4
            + [101]
            + [99] * 4
            + [8]
            + [99] * 4
            + [102, 0]
        )
        ret = {
            "input_ids": torch.tensor(ids),
            "pixel_values": torch.cat([image.feature, image.feature]),
            "image_grid_thw": torch.tensor([[1, 2, 8], [1, 4, 4]]),
            "pixel_values_videos": torch.cat([image.feature, image.feature]),
            "video_grid_thw": torch.tensor([[2, 2, 8]]),
        }
        tokens = self.ns["MultimodalSpecialTokens"](
            image_token_id=99, video_token_id=99
        )
        # Only the HF preprocessing boundary is a fixture. All SGLang
        # collection, placement, splitting and finalization methods are real.
        self.processor.process_mm_data = lambda **kwargs: ret
        raw = self.ns["BaseMultiModalProcessorOutput"](
            input_text="fixture", images=[object(), object()], videos=[object()]
        )
        items, input_ids, _ = self.processor.process_and_combine_mm_data(raw, tokens)
        self.assertEqual(input_ids.tolist(), ids)
        self.assertEqual(
            [item.offsets for item in items], [[(1, 4)], [(6, 9)], [(11, 14), (16, 19)]]
        )
        self.assertEqual(len({item.hash for item in items}), 3)
        single = self.finalize([self.item()])[0]
        self.assertEqual(single.hash, items[0].hash)

        offline = self.ns["BaseMultiModalProcessorOutput"](
            input_text="fixture",
            input_ids=ids,
            images=[
                {
                    "format": "processor_output",
                    "modality": "image",
                    "pixel_values": ret["pixel_values"],
                    "image_grid_thw": ret["image_grid_thw"],
                    "hash": 123,
                    "pad_value": 999,
                }
            ],
        )
        offline_items, offline_ids, _ = self.processor.process_and_combine_mm_data(
            offline, tokens
        )
        self.assertEqual(offline_ids.tolist(), ids)
        self.assertEqual([x.hash for x in offline_items], [x.hash for x in items[:2]])
        self.assertEqual(
            [x.pad_value for x in offline_items], [x.pad_value for x in items[:2]]
        )

    def test_real_vision_getters_forward_grid_and_position_ids_change(self):
        a, b = self.item(), self.item(((1, 4, 4),))
        captured = []

        class RecordingVision:
            dtype = torch.float32

            def __call__(self, pixels, *, grid_thw):
                captured.append((pixels.clone(), grid_thw.clone()))
                return pixels

        owner = SimpleNamespace(visual=RecordingVision(), use_data_parallel=False)
        model = self.ns["Glm5NextForConditionalGeneration"]
        model.get_image_feature(owner, [a])
        model.get_image_feature(owner, [b])
        self.assertTrue(torch.equal(captured[0][0], captured[1][0]))
        self.assertFalse(torch.equal(captured[0][1], captured[1][1]))
        video = self.item(((2, 2, 8),), video=True)
        model.get_video_feature(owner, [video])
        self.assertEqual(captured[-1][1].tolist(), [[1, 2, 8], [1, 2, 8]])

        # Test the real GLM spatial-position construction, with a sentinel
        # cosine/sine table. This still does not execute vision weights.
        positions = SimpleNamespace(
            spatial_merge_size=2,
            device=torch.device("cpu"),
            rotary_pos_emb=SimpleNamespace(
                get_cos_sin=lambda n: (
                    torch.arange(int(n)).unsqueeze(-1),
                    torch.zeros((int(n), 1)),
                )
            ),
        )
        rot = self.ns["Glm4vVisionModel"].rot_pos_emb
        ids_a = rot(positions, a.image_grid_thw)[2]
        ids_b = rot(positions, b.image_grid_thw)[2]
        self.assertEqual(ids_a.shape, ids_b.shape)
        self.assertFalse(torch.equal(ids_a, ids_b))

    def test_mixed_modalities_and_base_processor_identity_unchanged(self):
        image, video = self.finalize([self.item(), self.item(video=True)])
        self.assertNotEqual(image.hash, video.hash)
        plain = self.item()
        base = self.ns["BaseMultimodalProcessor"]()
        base.precompute_hash_before_cpu_transfer = True
        base.keep_mm_features_on_device = False
        base.use_cuda_ipc = False
        base._finalize_mm_items([plain], images=None)
        self.assertEqual(plain.hash, self.ns["hash_feature"](plain.feature))
        audio = self.item()
        audio.modality = self.ns["Modality"].AUDIO
        self.finalize([audio])
        self.assertEqual(audio.hash, plain.hash)

    def test_processor_output_ignores_unscoped_supplied_hash_and_pad(self):
        a, b = self.item(), self.item(((1, 4, 4),))
        for item in (a, b):
            item.format = self.ns["MultimodalInputFormat"].PROCESSOR_OUTPUT
            item.set_hash(123)
        self.finalize([a, b])
        self.assertNotEqual(a.hash, b.hash)
        fresh = self.finalize([self.item()])[0]
        self.assertEqual(a.hash, fresh.hash)
        self.assertEqual(a.pad_value, fresh.pad_value)

    def test_precomputed_embeddings_are_scoped_for_padding_too(self):
        # Scheduler bypasses vision for the actual precomputed_embeddings
        # field, but padding still participates in prefix-cache identity.
        a, b = self.item(), self.item(((1, 4, 4),))
        for item in (a, b):
            item.precomputed_embeddings = item.feature[:4].clone()
            item.feature = None
        self.finalize([a, b])
        self.assertNotEqual(a.hash, b.hash)
        result = self.ns["_get_precomputed_embedding"](
            [a], [0, 1], [0], [6], [a.offsets]
        )
        torch.testing.assert_close(result, a.precomputed_embeddings)

    def test_precomputed_format_with_feature_is_not_assumed_to_bypass_vision(self):
        a, b = self.item(), self.item(((1, 4, 4),))
        for item in (a, b):
            item.format = self.ns["MultimodalInputFormat"].PRECOMPUTED_EMBEDDING
        self.finalize([a, b])
        self.assertNotEqual(a.hash, b.hash)
        self.assertIsNone(
            self.ns["_get_precomputed_embedding"]([a], [0, 1], [0], [6], [a.offsets])
        )

    def test_hash_skip_option_still_disables_reuse(self):
        with patch.dict(os.environ, {"SGLANG_MM_SKIP_COMPUTE_HASH": "1"}):
            a, b = self.finalize([self.item(), self.item()])
        self.assertNotEqual(a.hash, b.hash)
        self.assertLess(a.hash, 1 << 64)
        self.assertLess(b.hash, 1 << 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
