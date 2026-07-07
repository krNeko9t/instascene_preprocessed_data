from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from instascene.manifest.io import (  # noqa: E402
    load_scene_paths_manifest,
    manifest_dataset_id,
    manifest_id_map_source,
    manifest_pair_by,
    resolve_scene_paths,
)
from instascene.manifest.models import ResolvedScenePaths  # noqa: E402
from instascene.manifest.sample import select_scene_subset  # noqa: E402
from instascene.scene.discovery.local import discover_local_extracted  # noqa: E402
from instascene.scene.loaders.scene import load_scene_from_paths  # noqa: E402
from instascene.scene.models import ScenePathRecord  # noqa: E402


class ManifestMetadataTests(unittest.TestCase):
    def test_manifest_helpers(self) -> None:
        payload = {
            "dataset_id": "3dovs",
            "dataset_root": "/data/3dovs",
            "id_map_source": "npy",
            "pair_by": "stem",
            "scenes": [
                {
                    "partition": "",
                    "scene_name": "bench",
                    "scene_root": "/data/3dovs/bench",
                    "image_dir": "/data/3dovs/bench/images",
                    "id_map_dir": "/data/3dovs/bench/id_maps",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            doc = load_scene_paths_manifest(path)
            self.assertEqual(manifest_dataset_id(doc), "3dovs")
            self.assertEqual(manifest_id_map_source(doc), "npy")
            self.assertEqual(manifest_pair_by(doc), "stem")


class SceneKeyTests(unittest.TestCase):
    def test_scene_key_without_partition(self) -> None:
        resolved = ResolvedScenePaths(
            partition="",
            scene_name="bench",
            scene_root=Path("/x/bench"),
            image_dir=Path("/x/bench/images"),
            id_map_dir=Path("/x/bench/id_maps"),
        )
        self.assertEqual(resolved.scene_key, "bench")

    def test_scene_key_with_partition(self) -> None:
        resolved = ResolvedScenePaths(
            partition="scene_001",
            scene_name="foo",
            scene_root=Path("/x/scene_001/foo"),
            image_dir=Path("/x/scene_001/foo/images"),
            id_map_dir=Path("/x/scene_001/foo/masks"),
        )
        self.assertEqual(resolved.scene_key, "scene_001/foo")


class SamplingTests(unittest.TestCase):
    def test_select_scene_subset_is_seed_stable(self) -> None:
        records = [
            ScenePathRecord("p", f"s{i}", Path(f"/s{i}"), Path(f"/s{i}/images"), Path(f"/s{i}/m"))
            for i in range(10)
        ]
        first = select_scene_subset(records, n_requested=3, seed=42, shuffle=False)
        second = select_scene_subset(records, n_requested=3, seed=42, shuffle=False)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


class LocalDiscoverAndLoadTests(unittest.TestCase):
    def test_discover_and_load_3dovs_bench(self) -> None:
        root = SCRIPTS.parent / "3dovs"
        if not root.is_dir():
            self.skipTest("local 3dovs data not available")
        records = discover_local_extracted(root)
        self.assertTrue(any(r.scene_name == "bench" for r in records))
        bench = next(r for r in records if r.scene_name == "bench")
        resolved = ResolvedScenePaths(
            partition=bench.partition,
            scene_name=bench.scene_name,
            scene_root=bench.scene_root,
            image_dir=bench.image_dir,
            id_map_dir=bench.id_map_dir,
            id_map_json=bench.id_map_json,
        )
        scene_data = load_scene_from_paths(
            "3dovs",
            resolved,
            id_map_source="npy",
            pair_by="stem",
        )
        self.assertGreater(len(scene_data.views), 0)
        self.assertGreater(len(scene_data.object_ids), 0)


if __name__ == "__main__":
    unittest.main()
