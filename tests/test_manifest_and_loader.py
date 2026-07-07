from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from instascene.manifest.io import load_scene_selection_manifest  # noqa: E402
from instascene.manifest.models import ResolvedScenePaths  # noqa: E402
from instascene.manifest.resolver import resolve_local_scene, resolve_scene_ref  # noqa: E402
from instascene.manifest.sample import select_scene_subset  # noqa: E402
from instascene.manifest.sample_registry import get_dataset_spec  # noqa: E402
from instascene.scene.discovery.local import discover_local_extracted  # noqa: E402
from instascene.scene.loaders.scene import load_scene_from_paths  # noqa: E402
from instascene.scene.ref import SceneRef  # noqa: E402


class ManifestLoadTests(unittest.TestCase):
    def test_load_scene_selection_manifest(self) -> None:
        payload = {
            "schema_version": 2,
            "kind": "scene_selection",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "sampling": {
                "num_candidates": 10,
                "num_selected": 1,
                "n_requested": 1,
                "seed": 42,
                "shuffle": False,
                "strict": False,
            },
            "dataset_roots": {"3dovs": "/data/3dovs"},
            "scenes": [{"dataset_id": "3dovs", "scene_id": "bench"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            doc = load_scene_selection_manifest(path)
            self.assertEqual(doc.schema_version, 2)
            self.assertEqual(doc.dataset_roots["3dovs"], Path("/data/3dovs"))
            self.assertEqual(doc.scenes[0].dataset_id, "3dovs")
            self.assertEqual(doc.scenes[0].scene_id, "bench")

    def test_dataset_spec_for_3dovs(self) -> None:
        spec = get_dataset_spec("3dovs")
        self.assertEqual(spec.id_map_source, "npy")
        self.assertEqual(spec.pair_by, "stem")


class SceneKeyTests(unittest.TestCase):
    def test_scene_key_is_scene_id(self) -> None:
        resolved = ResolvedScenePaths(
            scene_id="bench",
            scene_root=Path("/x/bench"),
            image_dir=Path("/x/bench/images"),
            id_map_dir=Path("/x/bench/id_maps"),
        )
        self.assertEqual(resolved.scene_key, "bench")

    def test_scene_key_with_partition_in_scene_id(self) -> None:
        resolved = ResolvedScenePaths(
            scene_id="scene_001/foo",
            scene_root=Path("/x/scene_001/foo"),
            image_dir=Path("/x/scene_001/foo/images"),
            id_map_dir=Path("/x/scene_001/foo/masks"),
        )
        self.assertEqual(resolved.scene_key, "scene_001/foo")


class SamplingTests(unittest.TestCase):
    def test_select_scene_subset_is_seed_stable(self) -> None:
        refs = [SceneRef(f"s{i}") for i in range(10)]
        first = select_scene_subset(refs, n_requested=3, seed=42, shuffle=False)
        second = select_scene_subset(refs, n_requested=3, seed=42, shuffle=False)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


class LocalDiscoverAndLoadTests(unittest.TestCase):
    def test_discover_and_load_3dovs_bench(self) -> None:
        root = SCRIPTS.parent / "3dovs"
        if not root.is_dir():
            self.skipTest("local 3dovs data not available")
        refs = discover_local_extracted(root)
        self.assertTrue(any(r.scene_id == "bench" for r in refs))
        resolved = resolve_local_scene(root, "bench")
        scene_data = load_scene_from_paths(
            "3dovs",
            resolved,
            id_map_source="npy",
            pair_by="stem",
        )
        self.assertGreater(len(scene_data.views), 0)
        self.assertGreater(len(scene_data.object_ids), 0)

    def test_resolve_scene_ref_matches_local_resolver(self) -> None:
        root = SCRIPTS.parent / "3dovs"
        if not root.is_dir():
            self.skipTest("local 3dovs data not available")
        resolved = resolve_scene_ref("3dovs", "bench", root)
        self.assertEqual(resolved.scene_id, "bench")
        self.assertTrue(resolved.image_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
