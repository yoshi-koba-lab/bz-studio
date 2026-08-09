from __future__ import annotations

import base64
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import tifffile

import mosaic


def _page_zip(stage_xyz):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for page in range(2):
            if stage_xyz is None:
                xml = "<Data><SingleFileProperty><Shooting /></SingleFileProperty></Data>"
            else:
                x, y, z = stage_xyz
                xml = (
                    "<Data><SingleFileProperty><Shooting>"
                    f"<StageLocationX>{x}</StageLocationX>"
                    f"<StageLocationY>{y}</StageLocationY>"
                    f"<StageLocationZ>{z}</StageLocationZ>"
                    "</Shooting></SingleFileProperty></Data>"
                )
            archive.writestr(str(page), xml)
    return payload.getvalue()


def _write_ome(path, planes, labels, colors, stage_xyz, pixel_yx=(3.0, 2.0)):
    data = np.stack(planes) if len(planes) > 1 else planes[0]
    dtype = np.dtype(data.dtype)
    height, width = planes[0].shape
    channels = "".join(
        f'<Channel ID="Channel:0:{i}" Name="{label}" Color="{color}" '
        'SamplesPerPixel="1" />'
        for i, (label, color) in enumerate(zip(labels, colors))
    )
    tiff_data = "".join(
        f'<TiffData IFD="{i}" FirstC="{i}" FirstZ="0" FirstT="0" PlaneCount="1" />'
        for i in range(len(planes))
    )
    annotation = ""
    if stage_xyz is not None:
        payload = _page_zip(stage_xyz)
        encoded = base64.b64encode(payload).decode("ascii")
        annotation = (
            '<StructuredAnnotations><FileAnnotation ID="Annotation:0">'
            '<Description>Metadata</Description><BinaryFile>'
            f'<BinData Compression="zlib" Length="{len(payload)}">{encoded}</BinData>'
            '</BinaryFile></FileAnnotation></StructuredAnnotations>'
        )
    physical = ""
    if pixel_yx[1] is not None:
        physical += f' PhysicalSizeX="{pixel_yx[1]}" PhysicalSizeXUnit="um"'
    if pixel_yx[0] is not None:
        physical += f' PhysicalSizeY="{pixel_yx[0]}" PhysicalSizeYUnit="um"'
    ome = (
        '<?xml version="1.0"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
        f'SizeX="{width}" SizeY="{height}" SizeC="{len(planes)}" SizeZ="1" SizeT="1" '
        f'Type="{dtype.name}"{physical}>'
        f"{channels}{tiff_data}</Pixels></Image>{annotation}</OME>"
    )
    tifffile.imwrite(path, data, photometric="minisblack", metadata=None, description=ome)


def _gci_xml(entries, rows=2, cols=2, compression=zipfile.ZIP_DEFLATED):
    archive_data = io.BytesIO()
    with zipfile.ZipFile(archive_data, "w", compression) as archive:
        archive.writestr(
            "GroupFileProperty/ImageJoint/properties.xml",
            "<Store><Enabled>True</Enabled>"
            f"<Row>{rows}</Row><Column>{cols}</Column></Store>",
        )
        archive.writestr(
            "GroupFileProperty/ImageList/OmeFileList/properties.xml",
            f"<Store><Count>{len(entries)}</Count></Store>",
        )
        for index, (row, col, file_path) in enumerate(entries):
            archive.writestr(
                f"GroupFileProperty/ImageList/OmeFileList/Index{index}/properties.xml",
                "<Store>"
                f"<IndexRow>{row}</IndexRow><IndexColumn>{col}</IndexColumn>"
                f"<FilePath>{file_path}</FilePath>"
                "</Store>",
            )
    return archive_data.getvalue()


def _make_fixture(
    root,
    missing=None,
    duplicate=False,
    mismatch=None,
    missing_stage=None,
    primary_missing_stage=None,
    conflicting_stage=None,
    different_z=None,
):
    entries = []
    files = {}
    order = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for number, (row, col) in enumerate(order, start=1):
        base = f"Image_{number:05d}"
        stage = (1_000_000 - col * 80_000, 2_000_000 - row * 60_000, 300_000)
        shape = (7, 8) if mismatch == (row, col) else (6, 8)
        ch1 = root / f"{base}_CH1.bz.ome.tif"
        chf = root / f"{base}_CHF.bz.ome.tif"
        overlay = root / f"{base}_Overlay.bz.ome.tif"
        _write_ome(
            ch1,
            [
                np.full(shape, number, np.uint8),
                np.full(shape, number + 10, np.uint8),
            ],
            ["Red", "Green"],
            [-16776961, 16711935],
            None if missing_stage == (row, col) or primary_missing_stage == (row, col)
            else stage,
        )
        chf_stage = None if missing_stage == (row, col) else stage
        if conflicting_stage == (row, col):
            chf_stage = (stage[0] + 1, stage[1], stage[2])
        elif different_z == (row, col):
            chf_stage = (stage[0], stage[1], stage[2] + 1)
        _write_ome(
            chf,
            [np.full(shape, number * 1000, np.uint16)],
            ["488"],
            [16711935],
            chf_stage,
        )
        tifffile.imwrite(overlay, np.zeros(shape, np.uint8))
        entries.extend([
            (row, col, ch1.name),
            (row, col, chf.name),
            (row, col, overlay.name),
        ])
        files[(row, col, "CH1")] = ch1
        files[(row, col, "CHF")] = chf
    if duplicate:
        duplicate_file = root / "Duplicate_CH1.bz.ome.tif"
        _write_ome(
            duplicate_file,
            [np.zeros((6, 8), np.uint8), np.zeros((6, 8), np.uint8)],
            ["Red", "Green"],
            [-16776961, 16711935],
            (1_000_000, 2_000_000, 300_000),
        )
        entries.append((0, 0, duplicate_file.name))
    (root / "Image.gci").write_bytes(_gci_xml(entries))
    if missing is not None:
        files[(*missing, "CHF")].unlink()
    return files


class MosaicReaderTests(unittest.TestCase):
    def test_generic_xy_child_uses_capture_folder_as_display_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "Section 7" / "XY01"
            root.mkdir(parents=True)
            _make_fixture(root)

            dataset = mosaic.load_mosaic_dataset(root)

            self.assertEqual(dataset.name, "Section 7")
            self.assertEqual(dataset.root, root.resolve())

    def test_loads_gci_sources_channels_stage_and_native_planes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "scan"
            root.mkdir()
            _make_fixture(root)
            progress = []

            dataset = mosaic.load_mosaic_dataset(
                root, progress=lambda message, fraction: progress.append((message, fraction))
            )

            self.assertEqual(dataset.root, root.resolve())
            self.assertEqual(dataset.name, "scan")
            self.assertEqual(dataset.grid_shape, (2, 2))
            self.assertEqual(dataset.tile_shape, (6, 8))
            self.assertEqual(dataset.dtype, np.dtype("uint16"))
            self.assertEqual(dataset.pixel_size_um_yx, (3.0, 2.0))
            self.assertEqual([(tile.row, tile.col) for tile in dataset.tiles],
                             [(0, 0), (0, 1), (1, 0), (1, 1)])
            self.assertTrue(all("Overlay" not in tile.files for tile in dataset.tiles))
            self.assertEqual(
                [(channel.key, channel.label, channel.color, channel.file_tag,
                  channel.page, channel.dtype) for channel in dataset.channels],
                [
                    ("CH1-p0", "Red", (255, 0, 0), "CH1", 0, np.dtype("uint8")),
                    ("CH1-p1", "Green", (0, 255, 0), "CH1", 1, np.dtype("uint8")),
                    ("CHF", "488", (0, 255, 0), "CHF", 0, np.dtype("uint16")),
                ],
            )
            self.assertEqual(
                mosaic.stage_origins(dataset),
                {(0, 0): (0.0, 0.0), (0, 1): (0.0, 40.0),
                 (1, 0): (20.0, 0.0), (1, 1): (20.0, 40.0)},
            )
            tile = next(tile for tile in dataset.tiles if (tile.row, tile.col) == (1, 1))
            np.testing.assert_array_equal(
                mosaic.read_plane(dataset, tile, "CH1-p1"),
                np.full((6, 8), 13, np.uint8),
            )
            np.testing.assert_array_equal(
                mosaic.read_plane(dataset, tile, "CHF"),
                np.full((6, 8), 3000, np.uint16),
            )
            self.assertEqual(progress[-1], ("Mosaic metadata ready", 1.0))
            self.assertTrue(any("mixed dtypes" in warning for warning in dataset.warnings))

    def test_missing_referenced_file_is_rejected_before_stitching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, missing=(0, 1))

            with self.assertRaisesRegex(
                    mosaic.MosaicMetadataError,
                    r"Incomplete mosaic source set.*Index\d+ \(0, 1\).*Restore"):
                mosaic.load_mosaic_dataset(root)

    def test_duplicate_source_tag_at_one_grid_position_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, duplicate=True)

            with self.assertRaisesRegex(mosaic.MosaicMetadataError, "Duplicate source tag"):
                mosaic.load_mosaic_dataset(root)

    def test_inconsistent_tile_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, mismatch=(1, 0))

            with self.assertRaisesRegex(mosaic.MosaicMetadataError, "Tile dimensions differ"):
                mosaic.load_mosaic_dataset(root)

    def test_missing_stage_is_warned_and_stage_origins_refuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, missing_stage=(1, 1))

            dataset = mosaic.load_mosaic_dataset(root)

            self.assertTrue(any("StageLocationX/Y metadata is missing" in warning
                                for warning in dataset.warnings))
            with self.assertRaisesRegex(mosaic.MosaicMetadataError, "StageLocationX/Y is missing"):
                mosaic.stage_origins(dataset)

    def test_stage_falls_back_to_another_source_for_the_same_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, primary_missing_stage=(1, 1))

            dataset = mosaic.load_mosaic_dataset(root)

            tile = next(tile for tile in dataset.tiles if (tile.row, tile.col) == (1, 1))
            self.assertIsNotNone(tile.stage_xyz)
            self.assertIn((1, 1), mosaic.stage_origins(dataset))
            self.assertTrue(any("StageLocationX/Y metadata is missing" in warning
                                for warning in dataset.warnings))

    def test_conflicting_stage_between_tile_sources_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, conflicting_stage=(1, 1))

            with self.assertRaisesRegex(
                mosaic.MosaicMetadataError, "inconsistent StageLocationX/Y values"
            ):
                mosaic.load_mosaic_dataset(root)

    def test_different_z_between_tile_sources_is_warned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _make_fixture(root, different_z=(1, 1))

            dataset = mosaic.load_mosaic_dataset(root)

            self.assertTrue(any("differing StageLocationZ" in warning
                                for warning in dataset.warnings))

    def test_corrupt_embedded_stage_member_becomes_a_warning(self):
        stage_xml = (
            b"<Data><StageLocationX>1000</StageLocationX>"
            b"<StageLocationY>2000</StageLocationY></Data>"
        )
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("0", stage_xml)
        corrupt = bytearray(payload.getvalue())
        offset = corrupt.index(b"<StageLocationX>")
        corrupt[offset + 1] = ord("X")
        encoded = base64.b64encode(corrupt).decode("ascii")
        ome = (
            "<OME><StructuredAnnotations><BinData>"
            f"{encoded}"
            "</BinData></StructuredAnnotations></OME>"
        )

        stage, warnings = mosaic._stage_from_ome(ome)

        self.assertIsNone(stage)
        self.assertTrue(any("ZIP member is unreadable" in warning
                            for warning in warnings))

    def test_partial_physical_size_cannot_hide_an_axis_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "Image_00001_CH1.bz.ome.tif"
            second = root / "Image_00002_CH1.bz.ome.tif"
            _write_ome(
                first, [np.zeros((6, 8), np.uint8)], ["CH1"], [-16776961],
                (1_000_000, 2_000_000, 0), pixel_yx=(3.0, None),
            )
            _write_ome(
                second, [np.zeros((6, 8), np.uint8)], ["CH1"], [-16776961],
                (999_000, 2_000_000, 0), pixel_yx=(4.0, 2.0),
            )
            entries = [(0, 0, first.name), (0, 1, second.name)]
            (root / "Image.gci").write_bytes(_gci_xml(entries, rows=1, cols=2))

            with self.assertRaisesRegex(
                mosaic.MosaicMetadataError, "inconsistent PhysicalSizeX/Y"
            ):
                mosaic.load_mosaic_dataset(root)

    def test_corrupt_gci_member_is_normalized_to_metadata_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = [(0, 0, "Image_00001_CH1.bz.ome.tif")]
            payload = bytearray(
                _gci_xml(entries, rows=1, cols=1, compression=zipfile.ZIP_STORED)
            )
            marker = b"<Enabled>True</Enabled>"
            offset = payload.index(marker)
            payload[offset + len("<Enabled>")] = ord("F")
            (root / "Image.gci").write_bytes(payload)

            with self.assertRaisesRegex(mosaic.MosaicMetadataError, "GCI XML is unreadable"):
                mosaic.load_mosaic_dataset(root)

    def test_stage_origins_preserve_fractional_pixel_prior(self):
        dataset = mosaic.MosaicDataset(
            root=Path("/scan"),
            name="scan",
            gci_path=Path("/scan/Image.gci"),
            tiles=(
                mosaic.MosaicTile(0, 0, 0, {}, (1_000_000, 2_000_000, 0)),
                mosaic.MosaicTile(1, 0, 1, {}, (999_999, 2_000_000, 0)),
            ),
            channels=(),
            tile_shape=(6, 8),
            dtype=np.dtype("uint8"),
            pixel_size_um_yx=(0.8, 0.8),
            grid_shape=(1, 2),
        )

        origins = mosaic.stage_origins(dataset)

        self.assertEqual(origins[(0, 0)], (0.0, 0.0))
        self.assertAlmostEqual(origins[(0, 1)][1], 0.00125, places=12)

    def test_folder_without_gci_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(mosaic.MosaicMetadataError, "No GCI file"):
                mosaic.load_mosaic_dataset(temp)


if __name__ == "__main__":
    unittest.main()
