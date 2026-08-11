from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("BZPS_TEST", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

import mosaic
import mosaic_engine as engine
import render


def _dataset(planes, origins, dtype="uint8", pixel_size=(0.8, 0.8)):
    positions = sorted(origins)
    shape = next(iter(planes.values())).shape
    channel = mosaic.MosaicChannel(
        "c", "Channel", None, "c", 0, np.dtype(dtype))
    tiles = tuple(
        mosaic.MosaicTile(i, p[0], p[1], {}, (0, 0, 0))
        for i, p in enumerate(positions)
    )
    return SimpleNamespace(
        channels=(channel,), dtype=np.dtype(dtype), tile_shape=shape, tiles=tiles,
        pixel_size_um_yx=pixel_size, root=Path("/fake"),
        gci_path=Path("/fake/Image.gci"), name="fake",
        grid_shape=(max(p[0] for p in positions) + 1,
                    max(p[1] for p in positions) + 1), warnings=(),
    )


class MosaicEngineTests(unittest.TestCase):
    def test_maximum_preview_quality_means_no_downsampling(self):
        geometry = SimpleNamespace(output_shape=(24000, 18000))
        dataset = SimpleNamespace(channels=(SimpleNamespace(key="c"),))
        downsample_values = []

        class FakeRenderer:
            def __init__(self, *_args, **kwargs):
                downsample_values.append(kwargs["downsample"])
                self.output_shape = (2, 3)
                self.output_dtype = np.dtype("uint8")

            def render_block(self, _y, _x, height, width):
                return np.zeros((height, width), np.uint8)

        with patch.object(engine, "ChunkRenderer", FakeRenderer):
            images, downsample = engine.render_preview_channels(
                dataset, geometry, ["c"], max_side=0,
                radiometric_correction=False)

        self.assertEqual(downsample, 1)
        self.assertEqual(downsample_values, [1])
        self.assertEqual(images["c"].shape, (2, 3))

    def test_striped_composite_matches_full_float_reference(self):
        rng = np.random.default_rng(3)
        images = {
            "a": rng.integers(0, 256, size=(120, 10000), dtype=np.uint8),
            "b": rng.integers(0, 256, size=(120, 10000), dtype=np.uint8),
        }
        views = [
            render.ChannelView("a", (54, 112, 255), 7, 233, 0.8),
            render.ChannelView("b", (238, 60, 74), 3, 247, 1.4),
        ]
        reference = np.zeros((120, 10000, 3), np.float32)
        for view in views:
            scaled = render.apply_levels(
                images[view.ch_id], view.lo, view.hi, view.gamma)
            for index, color in enumerate(view.color):
                reference[:, :, index] += scaled * (color / 255.0)
        reference = np.clip(reference * 255.0, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(render.composite(views, images), reference)

    def test_phase_sign_and_fractional_stage_delta(self):
        rng = np.random.default_rng(11)
        world = gaussian_filter(rng.normal(size=(500, 850)).astype(np.float32), 1.2)
        h, w = 320, 480
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        actual = (2.4, 296.3)
        stage = (0.37, 299.62)
        a = map_coordinates(world, [yy + 50, xx + 50], order=3, mode="reflect")
        b = map_coordinates(
            world, [yy + 50 + actual[0], xx + 50 + actual[1]],
            order=3, mode="reflect")

        edge = engine._measure_edge(
            a, b, (0, 0), (0, 1), stage, "x", max_shift=20)

        self.assertTrue(edge.accepted)
        np.testing.assert_allclose(edge.shift_yx, actual, atol=0.35)
        np.testing.assert_allclose(
            np.subtract(edge.shift_yx, edge.stage_delta_yx),
            edge.correction_yx, atol=1e-12)

    def test_global_inconsistent_edge_is_rejected_and_reported(self):
        origins = {(0, 0): (0.0, 0.0), (0, 1): (0.0, 100.0),
                   (1, 0): (80.0, 0.0), (1, 1): (80.0, 100.0)}
        planes = {p: np.zeros((128, 160), np.uint8) for p in origins}
        dataset = _dataset(planes, origins)

        def measured(_a, _b, pa, pb, stage, axis, _limit):
            shift = tuple(np.subtract(origins[pb], origins[pa]))
            if pa == (0, 0) and pb == (0, 1):
                shift = (shift[0] + 25.0, shift[1] - 20.0)
            return engine.RegistrationEdge(
                pa, pb, axis, tuple(stage), shift,
                tuple(np.subtract(shift, stage)), 0.9, 20.0, 0.5, 1.0,
                True, weight=1.0)

        with patch.object(mosaic, "stage_origins", return_value=origins), \
             patch.object(mosaic, "read_plane", side_effect=lambda d, t, k: planes[(t.row, t.col)]), \
             patch.object(engine, "_measure_edge", side_effect=measured):
            geometry = engine.estimate_geometry(dataset, "c")

        self.assertGreaterEqual(geometry.rejected_inconsistent_edges, 1)
        self.assertTrue(any(edge.reason == "global_inconsistent"
                            for edge in geometry.edges))
        self.assertLess(geometry.residual_max, 2.5)

    def test_geometry_extent_matches_fractional_pixel_support(self):
        plane = np.arange(16, dtype=np.uint8).reshape(4, 4)
        for delta, expected_shape in (
                ((0.4, 2.4), (4, 6)),
                ((0.6, 2.6), (5, 7))):
            origins = {(0, 0): (0.0, 0.0), (0, 1): delta}
            planes = {position: plane for position in origins}
            dataset = _dataset(planes, origins)

            def measured(_a, _b, pa, pb, stage, axis, _limit):
                shift = tuple(np.subtract(origins[pb], origins[pa]))
                return engine.RegistrationEdge(
                    pa, pb, axis, tuple(stage), shift, (0.0, 0.0),
                    0.95, 20.0, 0.5, 1.0, True, weight=1.0)

            with self.subTest(delta=delta), \
                    patch.object(mosaic, "stage_origins", return_value=origins), \
                    patch.object(
                        mosaic, "read_plane",
                        side_effect=lambda d, t, k: planes[(t.row, t.col)]), \
                    patch.object(engine, "_measure_edge", side_effect=measured):
                geometry = engine.estimate_geometry(dataset, "c")
                renderer = engine.ChunkRenderer(
                    dataset, geometry, "c", blend_mode="nearest")
                output = renderer.render_block(
                    0, 0, geometry.output_shape[0], geometry.output_shape[1])
                reduced_renderer = engine.ChunkRenderer(
                    dataset, geometry, "c", downsample=2,
                    blend_mode="nearest")
                reduced = reduced_renderer.render_block(
                    0, 0, reduced_renderer.output_shape[0],
                    reduced_renderer.output_shape[1])

            self.assertEqual(geometry.output_shape, expected_shape)
            self.assertEqual(output.shape, expected_shape)
            if delta[0] > 0.5:
                self.assertTrue(np.any(output[-1]))
            if delta[1] > 0.5:
                self.assertTrue(np.any(output[:, -1]))
            self.assertTrue(np.any(reduced[-1]))
            self.assertTrue(np.any(reduced[:, -1]))

    def test_fractional_renderer_is_block_invariant_and_nearest_is_native(self):
        origins = {(0, 0): (0.0, 0.0), (0, 1): (0.3, 30.7),
                   (1, 0): (31.4, 0.2), (1, 1): (31.7, 30.9)}
        yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
        planes = {
            p: (3 * (yy + oy) + 5 * (xx + ox) + 7).astype(np.float32)
            for p, (oy, ox) in origins.items()
        }
        dataset = _dataset(planes, origins, dtype="float32")
        geometry = engine.MosaicGeometry(
            origins, origins, "c", (96, 95))
        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: planes[(t.row, t.col)]):
            renderer = engine.ChunkRenderer(
                dataset, geometry, "c", blend_mode="feather",
                output_dtype=np.float32, radiometric_correction=False)
            whole = renderer.render_block(0, 0, 96, 95)
            tiled = np.zeros_like(whole)
            for y0 in range(0, 96, 17):
                for x0 in range(0, 95, 19):
                    block = renderer.render_block(y0, x0, 17, 19)
                    tiled[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
        np.testing.assert_array_equal(whole, tiled)

        checker = np.array([[17, 203, 17, 203], [203, 17, 203, 17],
                            [17, 203, 17, 203], [203, 17, 203, 17]], np.uint8)
        one_origin = {(0, 0): (0.5, 0.5)}
        one_dataset = _dataset({(0, 0): checker}, one_origin)
        one_geometry = engine.MosaicGeometry(one_origin, one_origin, "c", (5, 5))
        with patch.object(mosaic, "read_plane", return_value=checker):
            nearest = engine.ChunkRenderer(
                one_dataset, one_geometry, "c", blend_mode="nearest")
            output = nearest.render_block(0, 0, 5, 5)
        self.assertTrue(set(np.unique(output)).issubset({0, 17, 203}))
        self.assertFalse(any(value not in (0, 17, 203) for value in output.ravel()))

    def test_fractional_nearest_preserves_the_complete_outer_pixel_support(self):
        """Scientific nearest must not drop an edge at a fractional origin."""

        plane = np.arange(1, 17, dtype=np.uint8).reshape(4, 4)
        for origin, target_slice in (
                ((0.4, 0.4), (slice(0, 4), slice(0, 4))),
                ((0.6, 0.6), (slice(1, 5), slice(1, 5)))):
            origins = {(0, 0): origin}
            dataset = _dataset({(0, 0): plane}, origins)
            geometry = engine.MosaicGeometry(origins, origins, "c", (5, 5))
            expected = np.zeros((5, 5), np.uint8)
            expected[target_slice] = plane
            with patch.object(mosaic, "read_plane", return_value=plane):
                renderer = engine.ChunkRenderer(
                    dataset, geometry, "c", blend_mode="nearest")
                actual = renderer.render_block(0, 0, 5, 5)
                feather = engine.ChunkRenderer(
                    dataset, geometry, "c", blend_mode="feather",
                    output_dtype=np.float32, radiometric_correction=False)
                feathered = feather.render_block(0, 0, 5, 5)
            np.testing.assert_array_equal(actual, expected)
            support = np.zeros((5, 5), dtype=bool)
            support[target_slice] = True
            self.assertTrue(np.all(feathered[support] > 0))
            self.assertTrue(np.all(feathered[~support] == 0))

    def test_radiometric_initialization_reports_progress_and_cancels(self):
        tile_size, step = 32, 24
        origins = {(row, col): (float(row * step), float(col * step))
                   for row in range(6) for col in range(4)}
        planes = {position: np.full((tile_size, tile_size), 90, np.uint8)
                  for position in origins}
        dataset = _dataset(planes, origins)
        edges = []
        for position in origins:
            for neighbour, axis in (
                    ((position[0], position[1] + 1), "x"),
                    ((position[0] + 1, position[1]), "y")):
                if neighbour not in origins:
                    continue
                delta = tuple(np.subtract(origins[neighbour], origins[position]))
                edges.append(engine.RegistrationEdge(
                    position, neighbour, axis, delta, delta, (0.0, 0.0),
                    0.95, 20.0, 0.5, 1.0, True, weight=1.0,
                    residual_yx=(0.0, 0.0)))
        output_shape = (step * 5 + tile_size, step * 3 + tile_size)
        geometry = engine.MosaicGeometry(
            origins, origins, "c", output_shape, edges=edges)
        progress = []
        checks = [0]

        def cancel_during_dynamic_scan():
            checks[0] += 1
            return checks[0] >= 3

        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: planes[(t.row, t.col)]):
            with self.assertRaises(engine.MosaicCancelled):
                engine.render_preview_channels(
                    dataset, geometry, ["c"], max_side=256,
                    blend_mode="feather", radiometric_correction=True,
                    progress=lambda message, value: progress.append((message, value)),
                    cancel=cancel_during_dynamic_scan)

        self.assertTrue(progress)
        self.assertTrue(all(0.0 <= value <= 1.0 for _, value in progress))
        self.assertNotIn("c", geometry.spatial_models)
        self.assertNotIn("c", geometry.radiometric_models)

        preview_progress = []
        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: planes[(t.row, t.col)]):
            engine.render_preview_channels(
                dataset, geometry, ["c"], max_side=256,
                radiometric_correction=False,
                progress=lambda message, value: preview_progress.append(value))
        self.assertTrue(preview_progress)
        self.assertEqual(preview_progress[-1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(
            preview_progress, preview_progress[1:])))

    def test_radiometry_improves_seam_and_is_direction_invariant(self):
        rng = np.random.default_rng(5)
        texture = gaussian_filter(rng.normal(size=(600, 900)), 3)
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        world = (25 + 150 * texture + np.linspace(0, 25, 900)[None, :]).astype(np.float32)
        h = w = 512
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        origins = {(0, 0): (0.0, 0.0), (0, 1): (0.35, 300.4)}
        a = np.rint(map_coordinates(world, [yy, xx], order=1)).astype(np.uint8)
        b = np.rint(1.10 * map_coordinates(
            world, [yy + 0.35, xx + 300.4], order=1) + 7).clip(0, 255).astype(np.uint8)
        planes = {(0, 0): a, (0, 1): b}
        dataset = _dataset(planes, origins)

        def geometry(reverse=False):
            pa, pb = ((0, 1), (0, 0)) if reverse else ((0, 0), (0, 1))
            delta = tuple(np.subtract(origins[pb], origins[pa]))
            edge = engine.RegistrationEdge(
                pa, pb, "x", delta, delta, (0.0, 0.0), 0.9, 20.0,
                0.5, 1.0, True, weight=1.0, residual_yx=(0.0, 0.0))
            return engine.MosaicGeometry(
                dict(origins), dict(origins), "c", (513, 813), edges=[edge])

        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: planes[(t.row, t.col)]):
            forward = engine.estimate_radiometric_model(dataset, geometry(), "c")
            reverse = engine.estimate_radiometric_model(dataset, geometry(True), "c")

        self.assertEqual(forward.status, "applied")
        self.assertLess(forward.corrected_seam_bias, forward.raw_seam_bias * 0.20)
        for position in origins:
            self.assertAlmostEqual(forward.gains[position], reverse.gains[position], places=7)
            self.assertAlmostEqual(
                forward.offsets[position], reverse.offsets[position], delta=0.02)
            self.assertTrue(0.8 <= forward.gains[position] <= 1.25)

    def test_common_additive_field_is_split_validated_and_nearest_bypasses_it(self):
        rng = np.random.default_rng(29)
        count, tile_size, step = 10, 96, 64
        world_size = step * (count - 1) + tile_size
        world = gaussian_filter(
            rng.normal(size=(world_size, world_size)).astype(np.float32), 2.0)
        world = 85.0 + 32.0 * (world - world.min()) / (world.max() - world.min())
        yy, xx = np.mgrid[0:tile_size, 0:tile_size].astype(np.float32)
        local_y = yy / (tile_size - 1) - 0.5
        local_x = xx / (tile_size - 1) - 0.5
        truth = 4.0 * (1.6 * local_x ** 2 + 1.6 * local_y ** 2
                       + 0.30 * local_x + 0.25 * local_y)
        truth -= np.median(truth)
        origins = {(row, col): (float(row * step), float(col * step))
                   for row in range(count) for col in range(count)}
        planes = {}
        for position, (oy, ox) in origins.items():
            source = world[int(oy):int(oy) + tile_size,
                           int(ox):int(ox) + tile_size]
            planes[position] = np.rint(source + truth).clip(0, 255).astype(np.uint8)
        dataset = _dataset(planes, origins)
        edges = []
        for position in origins:
            for neighbour, axis in (
                    ((position[0], position[1] + 1), "x"),
                    ((position[0] + 1, position[1]), "y")):
                if neighbour not in origins:
                    continue
                delta = tuple(np.subtract(origins[neighbour], origins[position]))
                edges.append(engine.RegistrationEdge(
                    position, neighbour, axis, delta, delta, (0.0, 0.0),
                    0.95, 20.0, 0.5, 1.0, True, weight=1.0,
                    residual_yx=(0.0, 0.0)))
        geometry = engine.MosaicGeometry(
            origins, origins, "c", (world_size, world_size), edges=edges)

        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: planes[(t.row, t.col)]):
            model = engine.estimate_spatial_field_model(dataset, geometry, "c")

        self.assertEqual(model.status, "applied", model.warnings)
        self.assertLess(model.corrected_heldout_bias_dn,
                        model.raw_heldout_bias_dn * 0.80)
        self.assertLessEqual(model.new_clip_fraction, 1e-4)
        recovered = engine._resize_spatial_field(model.field_dn, truth.shape)
        self.assertGreater(np.corrcoef(recovered.ravel(), truth.ravel())[0, 1], 0.90)

        x_truth = 4.0 * (1.8 * local_x ** 2 + 0.35 * local_x)
        x_truth -= np.median(x_truth)
        x_world = 90.0 + 24.0 * (world - world.min()) / (world.max() - world.min())
        x_planes = {}
        for position, (oy, ox) in origins.items():
            source = x_world[int(oy):int(oy) + tile_size,
                             int(ox):int(ox) + tile_size]
            x_planes[position] = np.rint(source + x_truth).clip(0, 255).astype(np.uint8)
        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: x_planes[(t.row, t.col)]):
            x_model = engine.estimate_spatial_field_model(dataset, geometry, "c")
        self.assertEqual(x_model.status, "applied", x_model.warnings)
        self.assertLess(x_model.orientation_qc["x"]["corrected_bias_dn"],
                        x_model.orientation_qc["x"]["raw_bias_dn"] * 0.80)
        self.assertLessEqual(x_model.orientation_qc["y"]["corrected_p95_dn"],
                             x_model.orientation_qc["y"]["raw_p95_dn"] + 0.25)

        flat_planes = {position: np.full((tile_size, tile_size), 100, np.uint8)
                       for position in origins}
        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: flat_planes[(t.row, t.col)]):
            rejected = engine.estimate_spatial_field_model(dataset, geometry, "c")
        self.assertEqual(rejected.status, "identity")
        self.assertIsNone(rejected.field_dn)

        global_y, global_x = np.mgrid[0:world_size, 0:world_size].astype(np.float32)
        biology = (55.0 + 65.0 * np.exp(-(
            (global_y - 0.52 * world_size) ** 2
            + (global_x - 0.43 * world_size) ** 2) / (0.32 * world_size) ** 2)
            + 18.0 * global_x / world_size)
        biology_planes = {
            position: np.rint(biology[int(oy):int(oy) + tile_size,
                                       int(ox):int(ox) + tile_size]).astype(np.uint8)
            for position, (oy, ox) in origins.items()
        }
        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: biology_planes[(t.row, t.col)]):
            biology_model = engine.estimate_spatial_field_model(
                dataset, geometry, "c")
        self.assertEqual(biology_model.status, "identity")
        self.assertIsNone(biology_model.field_dn)

        geometry.spatial_models["c"] = model
        with patch.object(mosaic, "read_plane",
                          side_effect=lambda d, t, k: planes[(t.row, t.col)]), \
             patch.object(engine, "_cached_spatial_model",
                          side_effect=AssertionError("nearest estimated a field")):
            nearest = engine.ChunkRenderer(
                dataset, geometry, "c", blend_mode="nearest")
            output = nearest.render_block(0, 0, world_size, world_size)
        self.assertTrue(set(np.unique(output)).issubset(
            set().union(*(set(np.unique(value)) for value in planes.values())) | {0}))

    def test_low_information_channel_uses_heldout_offset_only_fallback(self):
        rng = np.random.default_rng(71)
        count, tile_size, step = 6, 96, 64
        world_size = step * (count - 1) + tile_size
        world = gaussian_filter(
            rng.normal(size=(world_size, world_size)).astype(np.float32), 4.0)
        world = np.rint(100.0 + world / max(1e-6, float(np.std(world))) * 0.35)
        origins = {(row, col): (float(row * step), float(col * step))
                   for row in range(count) for col in range(count)}
        # Only horizontal neighbours differ.  A global x/y median would hide
        # this behind the zero-bias vertical seams, so the directional gate is
        # part of the regression contract.
        tile_offsets = {position: float(position[1] % 3 - 1)
                        for position in origins}
        planes = {}
        for position, (oy, ox) in origins.items():
            source = world[int(oy):int(oy) + tile_size,
                           int(ox):int(ox) + tile_size]
            planes[position] = np.clip(
                source + tile_offsets[position], 0, 255).astype(np.uint8)
        dataset = _dataset(planes, origins)
        edges = []
        for position in origins:
            for neighbour, axis in (
                    ((position[0], position[1] + 1), "x"),
                    ((position[0] + 1, position[1]), "y")):
                if neighbour not in origins:
                    continue
                delta = tuple(np.subtract(origins[neighbour], origins[position]))
                edges.append(engine.RegistrationEdge(
                    position, neighbour, axis, delta, delta, (0.0, 0.0),
                    0.95, 20.0, 0.5, 1.0, True, weight=1.0,
                    residual_yx=(0.0, 0.0)))
        geometry = engine.MosaicGeometry(
            origins, origins, "c", (world_size, world_size), edges=edges)
        with patch.object(
                mosaic, "read_plane",
                side_effect=lambda d, t, k: planes[(t.row, t.col)]):
            model = engine.estimate_radiometric_model(dataset, geometry, "c")

        self.assertEqual(model.status, "applied", model.warnings)
        self.assertEqual(model.mode, "offset-only")
        self.assertEqual(model.validation_mode, "checkerboard-spatial-blocks")
        self.assertLess(model.corrected_seam_bias, model.raw_seam_bias * 0.90)
        self.assertLessEqual(model.corrected_seam_mad, model.raw_seam_mad * 1.02)
        self.assertLessEqual(model.new_clip_fraction, 1e-4)
        self.assertGreaterEqual(model.orientation_qc["x"]["raw_bias_dn"], 0.5)
        self.assertLessEqual(model.orientation_qc["y"]["raw_bias_dn"], 0.05)

    def test_pyramid_pixels_and_three_file_commit_rollback(self):
        h, w = 333, 33
        plane = ((np.arange(h)[:, None] * 7 + np.arange(w)[None, :] * 13) % 251).astype(np.uint8)
        origins = {(0, 0): (0.0, 0.0)}
        dataset = _dataset({(0, 0): plane}, origins)
        geometry = engine.MosaicGeometry(origins, origins, "c", (h, w))
        with tempfile.TemporaryDirectory() as tempdir, \
             patch.object(mosaic, "read_plane", return_value=plane), \
             patch.object(engine, "_pyramid_levels", return_value=[2, 4, 8, 16, 32]):
            path = Path(tempdir) / "mosaic.ome.tif"
            engine.write_pyramidal_ome(dataset, geometry, path, tile_size=144)
            with tifffile.TiffFile(path) as image:
                self.assertEqual(image.series[0].axes, "YX")
                self.assertEqual(len(image.series[0].levels), 6)
                np.testing.assert_array_equal(
                    image.series[0].levels[-1].asarray(),
                    engine._block_mean(plane, 32))
                self.assertEqual(image.pages[0].tilewidth, 160)

            sidecar = Path(str(path) + ".mosaic-qc.json")
            alignment = path.with_name(path.stem + "_alignment.csv")
            path.write_bytes(b"OLD_TIFF")
            sidecar.write_bytes(b"OLD_JSON")
            alignment.write_bytes(b"OLD_CSV")
            real_replace = engine.os.replace
            calls = []

            def fail_image_commit(source, target):
                calls.append((source, target))
                if len(calls) == 3:
                    raise OSError("injected TIFF commit failure")
                return real_replace(source, target)

            with patch.object(engine.os, "replace", side_effect=fail_image_commit):
                with self.assertRaises(engine.MosaicExportError):
                    engine.write_pyramidal_ome(dataset, geometry, path, tile_size=144)
            self.assertEqual(path.read_bytes(), b"OLD_TIFF")
            self.assertEqual(sidecar.read_bytes(), b"OLD_JSON")
            self.assertEqual(alignment.read_bytes(), b"OLD_CSV")

    def test_scientific_ome_defaults_to_native_nearest_without_radiometry(self):
        left = np.array([[11, 41, 11, 41], [41, 11, 41, 11],
                         [11, 41, 11, 41], [41, 11, 41, 11]], np.uint8)
        right = np.array([[73, 151, 73, 151], [151, 73, 151, 73],
                          [73, 151, 73, 151], [151, 73, 151, 73]], np.uint8)
        origins = {(0, 0): (0.4, 0.3), (0, 1): (0.6, 2.7)}
        planes = {(0, 0): left, (0, 1): right}
        dataset = _dataset(planes, origins)
        geometry = engine.MosaicGeometry(origins, origins, "c", (5, 7))
        native = set(np.unique(left)) | set(np.unique(right)) | {0}
        with tempfile.TemporaryDirectory() as tempdir, \
             patch.object(mosaic, "read_plane",
                          side_effect=lambda d, t, k: planes[(t.row, t.col)]), \
             patch.object(engine, "_cached_spatial_model",
                          side_effect=AssertionError("scientific OME estimated shading")):
            path = Path(tempdir) / "native.ome.tif"
            engine.write_pyramidal_ome(dataset, geometry, path)
            with tifffile.TiffFile(path) as image:
                base = image.series[0].asarray()
            qc = json.loads(
                Path(str(path) + ".mosaic-qc.json").read_text(encoding="utf-8"))
        self.assertTrue(set(np.unique(base)).issubset(native))
        self.assertEqual(qc["blend_mode"], "nearest")
        self.assertFalse(qc["radiometric_compensation"]["enabled_for_feather_only"])
        self.assertTrue(qc["radiometric_compensation"]["nearest_preserves_native_values"])

    def test_pdf_atomic_validation_and_scale_bar_overflow(self):
        image = Image.new("RGB", (120, 80), "white")
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "presentation.pdf"
            engine.atomic_save_presentation(image, path, "pdf")
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF-"))
            self.assertIn(b"%%EOF", data[-4096:])
            path.write_bytes(b"OLD_PDF")

            def corrupt_save(_self, target, *_args, **_kwargs):
                Path(target).write_bytes(b"not a PDF")

            with patch.object(Image.Image, "save", new=corrupt_save):
                with self.assertRaises(engine.MosaicExportError):
                    engine.atomic_save_presentation(image, path, "pdf")
            self.assertEqual(path.read_bytes(), b"OLD_PDF")

            png_path = Path(tempdir) / "presentation.png"
            open_modes = []
            real_open = Path.open

            def record_open(target, mode="r", *args, **kwargs):
                open_modes.append(mode)
                return real_open(target, mode, *args, **kwargs)

            with patch.object(Path, "open", new=record_open):
                engine.atomic_save_presentation(image, png_path, "png")
            self.assertIn("r+b", open_modes)

            # A cancellation received while Pillow is compressing must discard
            # the staged image rather than replacing a known-good destination.
            png_path.write_bytes(b"OLD_PNG")
            checks = [False, True]
            with self.assertRaises(engine.MosaicCancelled):
                engine.atomic_save_presentation(
                    image, png_path, "png",
                    cancel=lambda: checks.pop(0) if checks else True)
            self.assertEqual(png_path.read_bytes(), b"OLD_PNG")
            self.assertFalse(any(p.name.endswith(".part")
                                 for p in Path(tempdir).iterdir()))

        with self.assertRaisesRegex(ValueError, "exceeds available image width"):
            engine.draw_scale_bar(
                Image.new("RGB", (100, 50)), 1.0,
                engine.ScaleBarSpec(length_um=200, margin_px=5))

    def test_scale_bar_drag_anchor_is_resolution_independent(self):
        spec = engine.ScaleBarSpec(
            length_um=20, anchor_x=0.5, anchor_y=0.25,
            color=(0, 0, 0), thickness_px=4, margin_px=10,
            show_label=False, background="none")
        small = np.asarray(engine.draw_scale_bar(
            Image.new("RGB", (100, 60), "white"), 1.0, spec))
        large = np.asarray(engine.draw_scale_bar(
            Image.new("RGB", (200, 120), "white"), 1.0, spec))

        # At each output size the normalized anchor selects the same relative
        # point of the available top-left range, while physical length stays 20 px.
        np.testing.assert_array_equal(small[19, 40:60], 0)
        np.testing.assert_array_equal(large[34, 90:110], 0)

    def test_scale_bar_export_prefers_arial(self):
        loaded = object()
        with patch.object(engine.ImageFont, "truetype", return_value=loaded) as load:
            self.assertIs(engine._font(24), loaded)
        self.assertEqual(
            load.call_args.args[0],
            "/System/Library/Fonts/Supplemental/Arial.ttf")

    def test_pyramid_tile_generator_honours_cancellation_per_tile(self):
        array = np.arange(64, dtype=np.uint8).reshape(1, 8, 8)
        cancelled = [False]
        tiles = engine._level_tiles(array, 4, lambda: cancelled[0])
        np.testing.assert_array_equal(next(tiles), array[0, :4, :4])
        cancelled[0] = True
        with self.assertRaises(engine.MosaicCancelled):
            next(tiles)


if __name__ == "__main__":
    unittest.main()
