"""Read GCI-described OME-TIFF XY scans without Qt or export side effects."""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
import zlib
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image


class MosaicError(Exception):
    """Base error for a mosaic input that cannot be represented safely."""


class MosaicMetadataError(MosaicError):
    """Required GCI or OME metadata is absent or contradictory."""


@dataclass(frozen=True)
class MosaicChannel:
    key: str
    label: str
    color: Optional[Tuple[int, int, int]]
    file_tag: str
    page: int
    dtype: np.dtype


@dataclass(frozen=True)
class MosaicTile:
    index: int
    row: int
    col: int
    files: Mapping[str, Path]
    # KEYENCE StageLocation values are integer nanometres in the embedded page XML.
    stage_xyz: Optional[Tuple[int, int, Optional[int]]]


@dataclass(frozen=True)
class MosaicDataset:
    root: Path
    name: str
    gci_path: Path
    tiles: Tuple[MosaicTile, ...]
    channels: Tuple[MosaicChannel, ...]
    tile_shape: Tuple[int, int]
    # Mixed source depths are legitimate; this is the lossless common output dtype.
    dtype: np.dtype
    pixel_size_um_yx: Tuple[Optional[float], Optional[float]]
    grid_shape: Tuple[int, int]
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _GciEntry:
    index: int
    row: int
    col: int
    file_path: str


@dataclass
class _TileBuilder:
    index: int
    row: int
    col: int
    files: Dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class _OmeInfo:
    channels: Tuple[MosaicChannel, ...]
    tile_shape: Tuple[int, int]
    pixel_size_um_yx: Tuple[Optional[float], Optional[float]]
    ome_xml: str
    warnings: Tuple[str, ...]


_INDEX_MEMBER = re.compile(
    r"(?:^|/)GroupFileProperty/ImageList/OmeFileList/Index(\d+)/properties\.xml$"
)
_OME_SUFFIX = re.compile(r"(?:\.bz)?\.ome\.tiff?$", re.IGNORECASE)


def _report(progress: Optional[Callable], message: str, fraction: float) -> None:
    if progress is not None:
        progress(message, fraction)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _one_text(root: ET.Element, name: str) -> Optional[str]:
    for element in root.iter():
        if _local(element.tag) == name:
            text = (element.text or "").strip()
            return text or None
    return None


def _required_int(root: ET.Element, name: str, context: str) -> int:
    raw = _one_text(root, name)
    try:
        return int(raw) if raw is not None else int("")
    except ValueError as exc:
        raise MosaicMetadataError(f"{context}: {name} is missing or invalid") from exc


def _find_member(names: Sequence[str], suffix: str) -> str:
    matches = [name for name in names if name.rstrip("/").endswith(suffix)]
    if len(matches) != 1:
        raise MosaicMetadataError(
            f"GCI must contain exactly one {suffix} (found {len(matches)})"
        )
    return matches[0]


def _read_xml(archive: zipfile.ZipFile, member: str) -> ET.Element:
    try:
        return ET.fromstring(archive.read(member))
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise MosaicMetadataError(f"GCI XML is unreadable: {member}") from exc


def _resolve_gci(path: Path) -> Tuple[Path, Path]:
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".gci":
            raise MosaicMetadataError(f"Not a GCI file: {path}")
        return path.parent.resolve(), path.resolve()
    if not path.is_dir():
        raise MosaicMetadataError(f"Mosaic folder does not exist: {path}")
    candidates = sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() == ".gci"
    )
    if not candidates:
        raise MosaicMetadataError(f"No GCI file found directly in {path}")
    if len(candidates) > 1:
        raise MosaicMetadataError(f"Multiple GCI files found directly in {path}")
    return path.resolve(), candidates[0].resolve()


def _gci_entries(gci_path: Path) -> Tuple[Tuple[int, int], list[_GciEntry], list[str]]:
    warnings = []
    try:
        archive = zipfile.ZipFile(gci_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise MosaicMetadataError(f"GCI is not a readable ZIP archive: {gci_path}") from exc
    with archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise MosaicMetadataError("GCI contains duplicate archive member names")
        joint_member = _find_member(
            names, "GroupFileProperty/ImageJoint/properties.xml"
        )
        joint = _read_xml(archive, joint_member)
        enabled = (_one_text(joint, "Enabled") or "").lower()
        rows = _required_int(joint, "Row", "ImageJoint")
        cols = _required_int(joint, "Column", "ImageJoint")
        if enabled != "true" or rows <= 0 or cols <= 0:
            raise MosaicMetadataError(
                "GCI ImageJoint is not an enabled XY grid with positive Row/Column"
            )

        count_member = _find_member(
            names, "GroupFileProperty/ImageList/OmeFileList/properties.xml"
        )
        declared_count = _required_int(
            _read_xml(archive, count_member), "Count", "OmeFileList"
        )
        index_members = []
        for name in names:
            match = _INDEX_MEMBER.search(name)
            if match:
                index_members.append((int(match.group(1)), name))
        index_members.sort()
        if declared_count != len(index_members):
            warnings.append(
                f"GCI declares {declared_count} OmeFileList entries but contains "
                f"{len(index_members)} Index XML records"
            )
        if len({index for index, _ in index_members}) != len(index_members):
            raise MosaicMetadataError("GCI contains duplicate OmeFileList Index values")

        entries = []
        for index, member in index_members:
            props = _read_xml(archive, member)
            row = _required_int(props, "IndexRow", f"OmeFileList Index{index}")
            col = _required_int(props, "IndexColumn", f"OmeFileList Index{index}")
            file_path = _one_text(props, "FilePath")
            if not file_path:
                raise MosaicMetadataError(
                    f"OmeFileList Index{index}: FilePath is missing"
                )
            if not (0 <= row < rows and 0 <= col < cols):
                raise MosaicMetadataError(
                    f"OmeFileList Index{index}: ({row}, {col}) is outside "
                    f"the declared {rows} x {cols} grid"
                )
            entries.append(_GciEntry(index, row, col, file_path))
    if not entries:
        raise MosaicMetadataError("GCI OmeFileList contains no Index records")
    return (rows, cols), entries, warnings


def _source_tag(file_path: str) -> str:
    name = Path(file_path.replace("\\", "/")).name
    base = _OME_SUFFIX.sub("", name)
    if base == name:
        base = re.sub(r"\.tiff?$", "", name, flags=re.IGNORECASE)
    tag = base.rsplit("_", 1)[-1].strip()
    if not tag:
        raise MosaicMetadataError(f"Cannot determine source tag from {file_path}")
    return tag


def _source_path(root: Path, raw: str) -> Path:
    normalised = raw.replace("\\", "/")
    relative = Path(normalised)
    if relative.is_absolute() or ".." in relative.parts:
        raise MosaicMetadataError(f"GCI FilePath escapes the dataset folder: {raw}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MosaicMetadataError(f"GCI FilePath escapes the dataset folder: {raw}") from exc
    return resolved


def _decode_color(raw: Optional[str]) -> Optional[Tuple[int, int, int]]:
    try:
        value = int(raw) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    red, green, blue = (value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF
    return (red, green, blue) if red or green or blue else None


def _unit_to_um(value: Optional[str], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    key = (unit or "µm").strip().lower().replace("μ", "µ")
    factors = {
        "µm": 1.0,
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 0.001,
        "mm": 1000.0,
        "m": 1_000_000.0,
    }
    factor = factors.get(key)
    if factor is None or not math.isfinite(number) or number <= 0:
        return None
    return number * factor


def _ome_images(root: ET.Element) -> list[ET.Element]:
    return [element for element in root if _local(element.tag) == "Image"]


def _pixels_of(image: Optional[ET.Element]) -> Optional[ET.Element]:
    if image is None:
        return None
    return next(
        (element for element in image if _local(element.tag) == "Pixels"), None
    )


def _page_coords(series, ordinal: int) -> Dict[str, int]:
    axes = str(series.axes)
    dims = [(axis, int(series.shape[i])) for i, axis in enumerate(axes)
            if axis not in ("Y", "X", "S")]
    if not dims:
        return {}
    count = math.prod(size for _, size in dims)
    if count != len(series.pages):
        return {}
    coords = np.unravel_index(ordinal, tuple(size for _, size in dims))
    return {axis: int(coords[i]) for i, (axis, _) in enumerate(dims)}


def _inspect_ome(path: Path, file_tag: str) -> _OmeInfo:
    warnings = []
    try:
        with tifffile.TiffFile(path) as image:
            ome_xml = image.ome_metadata or ""
            if not ome_xml:
                raise MosaicMetadataError(f"OME metadata is missing: {path.name}")
            try:
                ome = ET.fromstring(ome_xml)
            except ET.ParseError as exc:
                raise MosaicMetadataError(f"OME XML is invalid: {path.name}") from exc
            images = _ome_images(ome)
            page_specs = []
            sizes = []
            physical = []
            seen_pages = set()
            for series_index, series in enumerate(image.series):
                axes = str(series.axes)
                if "S" in axes and int(series.shape[axes.index("S")]) > 1:
                    raise MosaicMetadataError(
                        f"Packed RGB samples are unsupported in {path.name}"
                    )
                pixels = _pixels_of(images[series_index] if series_index < len(images) else None)
                channel_meta = [] if pixels is None else [
                    element for element in pixels if _local(element.tag) == "Channel"
                ]
                if pixels is None:
                    warnings.append(f"{path.name}: OME Pixels metadata is missing")
                else:
                    py = _unit_to_um(
                        pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit")
                    )
                    px = _unit_to_um(
                        pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit")
                    )
                    physical.append((py, px))
                for ordinal, page in enumerate(series.pages):
                    shape = tuple(int(value) for value in page.shape)
                    if len(shape) != 2:
                        raise MosaicMetadataError(
                            f"OME page is not a 2-D plane in {path.name}: {shape}"
                        )
                    sizes.append(shape)
                    page_index = int(page.index)
                    if page_index in seen_pages:
                        continue
                    seen_pages.add(page_index)
                    coords = _page_coords(series, ordinal)
                    channel_index = coords.get("C", 0)
                    meta = channel_meta[channel_index] if channel_index < len(channel_meta) else None
                    name = (meta.get("Name") or "").strip() if meta is not None else ""
                    color = _decode_color(meta.get("Color") if meta is not None else None)
                    suffix = ", ".join(
                        f"{axis}={value + 1}" for axis, value in coords.items()
                        if axis != "C"
                    )
                    if series_index:
                        suffix = ", ".join(filter(None, (f"series={series_index + 1}", suffix)))
                    page_specs.append((page_index, name, color, np.dtype(page.dtype), suffix))
    except MosaicError:
        raise
    except (OSError, RuntimeError, tifffile.TiffFileError, ValueError) as exc:
        raise MosaicMetadataError(f"Cannot read OME-TIFF {path.name}: {exc}") from exc

    if not page_specs:
        raise MosaicMetadataError(f"OME-TIFF contains no readable pages: {path.name}")
    if len(set(sizes)) != 1:
        raise MosaicMetadataError(f"OME pages have different dimensions in {path.name}")
    known_y = {pair[0] for pair in physical if pair[0] is not None}
    known_x = {pair[1] for pair in physical if pair[1] is not None}
    if len(known_y) > 1 or len(known_x) > 1:
        raise MosaicMetadataError(f"OME series have inconsistent pixel sizes in {path.name}")
    pixel_size = (
        next(iter(known_y)) if known_y else None,
        next(iter(known_x)) if known_x else None,
    )
    if pixel_size == (None, None):
        warnings.append(f"{path.name}: PhysicalSizeX/Y metadata is missing or invalid")
    elif None in pixel_size or any(None in pair for pair in physical):
        warnings.append(f"{path.name}: some OME series lack PhysicalSizeX/Y")

    multiple = len(page_specs) > 1
    channels = []
    for page, name, color, dtype, suffix in sorted(page_specs):
        key = file_tag if not multiple else f"{file_tag}-p{page}"
        label = name or (file_tag if not multiple else f"{file_tag} [{page + 1}]")
        if suffix:
            label = f"{label} ({suffix})"
        channels.append(MosaicChannel(key, label, color, file_tag, page, dtype))
    return _OmeInfo(tuple(channels), sizes[0], pixel_size, ome_xml, tuple(warnings))


def _zip_payload(raw: bytes) -> Optional[zipfile.ZipFile]:
    try:
        return zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        try:
            return zipfile.ZipFile(io.BytesIO(zlib.decompress(raw)))
        except (zlib.error, zipfile.BadZipFile):
            return None


def _stage_from_ome(ome_xml: str) -> Tuple[Optional[Tuple[int, int, Optional[int]]], list[str]]:
    warnings = []
    try:
        ome = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None, ["OME XML cannot be parsed for stage metadata"]
    locations = []
    for annotations in ome.iter():
        if _local(annotations.tag) != "StructuredAnnotations":
            continue
        for element in annotations.iter():
            if _local(element.tag) != "BinData" or not (element.text or "").strip():
                continue
            try:
                payload = base64.b64decode("".join(element.text.split()), validate=True)
            except (ValueError, binascii.Error):
                continue
            archive = _zip_payload(payload)
            if archive is None:
                continue
            with archive:
                try:
                    members = sorted(
                        (name for name in archive.namelist()
                         if Path(name).name.isdigit()),
                        key=lambda name: int(Path(name).name),
                    )
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    warnings.append("OME page metadata ZIP is unreadable")
                    continue
                for member in members:
                    try:
                        page = ET.fromstring(archive.read(member))
                    except (
                        KeyError, OSError, RuntimeError, zipfile.BadZipFile, ET.ParseError
                    ):
                        warnings.append(
                            f"OME page metadata ZIP member is unreadable: {member}"
                        )
                        continue
                    raw_x = _one_text(page, "StageLocationX")
                    raw_y = _one_text(page, "StageLocationY")
                    raw_z = _one_text(page, "StageLocationZ")
                    if raw_x is None or raw_y is None:
                        continue
                    try:
                        location = (
                            int(raw_x), int(raw_y), int(raw_z) if raw_z is not None else None
                        )
                    except ValueError:
                        continue
                    locations.append(location)
    if not locations:
        return None, warnings + [
            "StageLocationX/Y metadata is missing from OME page XML"
        ]
    first = locations[0]
    if any(location != first for location in locations[1:]):
        warnings.append("OME page XML contains inconsistent StageLocation values; first used")
    return first, warnings


def _same_channel_structure(left: _OmeInfo, right: _OmeInfo) -> bool:
    return [
        (channel.page, channel.dtype) for channel in left.channels
    ] == [
        (channel.page, channel.dtype) for channel in right.channels
    ]


def load_mosaic_dataset(path, progress=None) -> MosaicDataset:
    """Load one GCI-described XY scan and index only its existing source OME files."""
    root, gci_path = _resolve_gci(Path(path))
    _report(progress, "Reading GCI metadata…", 0.02)
    grid_shape, entries, warnings = _gci_entries(gci_path)

    builders: Dict[Tuple[int, int], _TileBuilder] = {}
    tag_order: Dict[str, int] = {}
    declared_tags = set()
    missing_sources = []
    for entry in entries:
        tag = _source_tag(entry.file_path)
        if tag.casefold() == "overlay":
            continue
        declared_tags.add(tag)
        source = _source_path(root, entry.file_path)
        if not source.is_file():
            missing_sources.append(
                f"Index{entry.index} ({entry.row}, {entry.col}) {entry.file_path}")
            continue
        key = (entry.row, entry.col)
        builder = builders.setdefault(
            key, _TileBuilder(entry.index, entry.row, entry.col)
        )
        builder.index = min(builder.index, entry.index)
        if tag in builder.files:
            raise MosaicMetadataError(
                f"Duplicate source tag {tag!r} at grid position ({entry.row}, {entry.col})"
            )
        builder.files[tag] = source
        tag_order.setdefault(tag, entry.index)
    if not builders:
        raise MosaicMetadataError("GCI references no existing non-Overlay source files")
    if missing_sources:
        shown = "; ".join(missing_sources[:6])
        suffix = f"; and {len(missing_sources) - 6} more" if len(missing_sources) > 6 else ""
        raise MosaicMetadataError(
            "Incomplete mosaic source set; referenced file(s) are missing: "
            f"{shown}{suffix}. Restore the files before stitching."
        )

    expected = grid_shape[0] * grid_shape[1]
    if len(builders) != expected:
        raise MosaicMetadataError(
            f"ImageJoint declares {expected} grid positions but only {len(builders)} "
            "contain existing source files"
        )

    ordered_builders = sorted(builders.values(), key=lambda item: (item.row, item.col))
    all_sources = [(builder, tag, source)
                   for builder in ordered_builders
                   for tag, source in builder.files.items()]
    infos: Dict[Path, _OmeInfo] = {}
    canonical: Dict[str, _OmeInfo] = {}
    tile_shape = None
    known_pixel_y = set()
    known_pixel_x = set()
    dtypes = []
    total = len(all_sources)
    for position, (builder, tag, source) in enumerate(all_sources, start=1):
        _report(
            progress,
            f"Reading OME metadata ({position}/{total}): {source.name}",
            0.05 + 0.75 * position / max(1, total),
        )
        info = _inspect_ome(source, tag)
        infos[source] = info
        warnings.extend(info.warnings)
        if tile_shape is None:
            tile_shape = info.tile_shape
        elif info.tile_shape != tile_shape:
            raise MosaicMetadataError(
                f"Tile dimensions differ: {source.name} is {info.tile_shape}, "
                f"expected {tile_shape}"
            )
        if info.pixel_size_um_yx[0] is not None:
            known_pixel_y.add(info.pixel_size_um_yx[0])
        if info.pixel_size_um_yx[1] is not None:
            known_pixel_x.add(info.pixel_size_um_yx[1])
        dtypes.extend(channel.dtype for channel in info.channels)
        previous = canonical.get(tag)
        if previous is None:
            canonical[tag] = info
        elif not _same_channel_structure(previous, info):
            raise MosaicMetadataError(
                f"OME page/dtype structure differs for source tag {tag}: {source.name}"
            )
        elif [channel.label for channel in previous.channels] != [
                channel.label for channel in info.channels]:
            warnings.append(
                f"{source.name}: channel labels differ from the first {tag} source"
            )
        elif [channel.color for channel in previous.channels] != [
                channel.color for channel in info.channels]:
            warnings.append(
                f"{source.name}: channel colors differ from the first {tag} source"
            )
    if len(known_pixel_y) > 1 or len(known_pixel_x) > 1:
        raise MosaicMetadataError("Source OME files have inconsistent PhysicalSizeX/Y")
    pixel_size = (
        next(iter(known_pixel_y)) if known_pixel_y else None,
        next(iter(known_pixel_x)) if known_pixel_x else None,
    )

    tiles = []
    all_tags = set(canonical)
    if all_tags != declared_tags:
        missing = ", ".join(sorted(declared_tags - all_tags))
        raise MosaicMetadataError(
            f"No readable source files were found for required tag(s): {missing}"
        )
    for builder in ordered_builders:
        missing_tags = sorted(all_tags - set(builder.files))
        if missing_tags:
            raise MosaicMetadataError(
                f"Tile ({builder.row}, {builder.col}) lacks source tag(s): "
                f"{', '.join(missing_tags)}"
            )
        source_stages = []
        for source in builder.files.values():
            source_stage, stage_warnings = _stage_from_ome(infos[source].ome_xml)
            warnings.extend(
                f"Tile ({builder.row}, {builder.col}), {source.name}: {warning}"
                for warning in stage_warnings
            )
            if source_stage is not None:
                source_stages.append((source, source_stage))
        if source_stages:
            stage = source_stages[0][1]
            conflicts = [
                source.name for source, source_stage in source_stages[1:]
                if source_stage[:2] != stage[:2]
            ]
            if conflicts:
                raise MosaicMetadataError(
                    f"Tile ({builder.row}, {builder.col}) source files have "
                    f"inconsistent StageLocationX/Y values: "
                    f"{source_stages[0][0].name}, {', '.join(conflicts)}"
                )
            if any(source_stage[2] != stage[2]
                   for _, source_stage in source_stages[1:]):
                warnings.append(
                    f"Tile ({builder.row}, {builder.col}) source files have differing "
                    "StageLocationZ values; first used"
                )
        else:
            stage = None
        tiles.append(MosaicTile(
            builder.index, builder.row, builder.col, dict(builder.files), stage
        ))

    channels = []
    for tag in sorted(canonical, key=lambda value: (tag_order[value], value)):
        channels.extend(canonical[tag].channels)
    keys = [channel.key for channel in channels]
    if len(keys) != len(set(keys)):
        raise MosaicMetadataError("Source channel keys are not unique")
    promoted_dtype = np.dtype(np.result_type(*dtypes))
    if len({np.dtype(dtype).str for dtype in dtypes}) > 1:
        warnings.append(
            f"Source channels use mixed dtypes; dataset dtype promoted to {promoted_dtype}"
        )
    _report(progress, "Mosaic metadata ready", 1.0)
    # Acquisition software commonly stores the manifest in a generic ``XY01``
    # child.  Use the parent capture name in the UI/export while keeping the
    # exact resolved root as the immutable data identity.
    display_name = (root.parent.name
                    if re.fullmatch(r"XY\d+", root.name, re.IGNORECASE)
                    and root.parent != root else root.name)
    return MosaicDataset(
        root=root,
        name=display_name,
        gci_path=gci_path,
        tiles=tuple(tiles),
        channels=tuple(channels),
        tile_shape=tile_shape or (0, 0),
        dtype=promoted_dtype,
        pixel_size_um_yx=pixel_size,
        grid_shape=grid_shape,
        warnings=tuple(warnings),
    )


def read_plane(dataset: MosaicDataset, tile: MosaicTile, channel_key: str) -> np.ndarray:
    """Read one native-depth 2-D source page selected by dataset channel key."""
    channel = next(
        (candidate for candidate in dataset.channels if candidate.key == channel_key), None
    )
    if channel is None:
        raise KeyError(f"Unknown mosaic channel: {channel_key}")
    source = tile.files.get(channel.file_tag)
    if source is None:
        raise MosaicMetadataError(
            f"Tile ({tile.row}, {tile.col}) has no {channel.file_tag} source"
        )
    try:
        with tifffile.TiffFile(source) as image:
            if not (0 <= channel.page < len(image.pages)):
                raise MosaicMetadataError(
                    f"OME page {channel.page} is missing from {source.name}"
                )
            plane = np.asarray(image.pages[channel.page].asarray())
    except MosaicError:
        raise
    except (OSError, RuntimeError, tifffile.TiffFileError, ValueError) as exc:
        # Pillow decodes the LZW files shipped by the instrument without imagecodecs.
        try:
            with Image.open(source) as image:
                image.seek(channel.page)
                plane = np.asarray(image).copy()
        except Exception as fallback_exc:
            raise MosaicMetadataError(
                f"Cannot read {source.name}: {fallback_exc}"
            ) from exc
    if plane.ndim != 2 or plane.shape != dataset.tile_shape:
        raise MosaicMetadataError(
            f"Plane shape changed in {source.name}: {plane.shape}, "
            f"expected {dataset.tile_shape}"
        )
    if plane.dtype != channel.dtype:
        raise MosaicMetadataError(
            f"Plane dtype changed in {source.name}: {plane.dtype}, expected {channel.dtype}"
        )
    return plane


def _grid_stage_differences(
        tiles: Sequence[MosaicTile], horizontal: bool) -> list[int]:
    by_position = {(tile.row, tile.col): tile for tile in tiles}
    differences = []
    for tile in tiles:
        neighbour = by_position.get(
            (tile.row, tile.col + 1) if horizontal else (tile.row + 1, tile.col)
        )
        if neighbour is None or tile.stage_xyz is None or neighbour.stage_xyz is None:
            continue
        axis = 0 if horizontal else 1
        differences.append(neighbour.stage_xyz[axis] - tile.stage_xyz[axis])
    return differences


def _stage_sign(tiles: Sequence[MosaicTile], horizontal: bool) -> int:
    differences = _grid_stage_differences(tiles, horizontal)
    if not differences:
        return -1
    nonzero = [difference for difference in differences if difference]
    if not nonzero:
        axis = "X/column" if horizontal else "Y/row"
        raise MosaicMetadataError(f"Stage {axis} coordinates do not change across the grid")
    if min(nonzero) < 0 < max(nonzero):
        axis = "X/column" if horizontal else "Y/row"
        raise MosaicMetadataError(f"Stage {axis} direction is inconsistent across the grid")
    return 1 if float(np.median(nonzero)) > 0 else -1


def stage_origins(dataset: MosaicDataset) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Convert nanometre stage locations to normalized initial pixel origins."""
    pixel_y, pixel_x = dataset.pixel_size_um_yx
    if pixel_y is None or pixel_x is None or pixel_y <= 0 or pixel_x <= 0:
        raise MosaicMetadataError("PhysicalSizeX/Y is required for stage pixel origins")
    missing = [tile for tile in dataset.tiles
               if tile.stage_xyz is None or None in tile.stage_xyz[:2]]
    if missing:
        shown = ", ".join(f"({tile.row}, {tile.col})" for tile in missing[:8])
        raise MosaicMetadataError(f"StageLocationX/Y is missing for tile(s): {shown}")
    sign_x = _stage_sign(dataset.tiles, horizontal=True)
    sign_y = _stage_sign(dataset.tiles, horizontal=False)
    raw = {}
    for tile in dataset.tiles:
        stage_x, stage_y, _ = tile.stage_xyz
        raw[(tile.row, tile.col)] = (
            sign_y * stage_y / (1000.0 * pixel_y),
            sign_x * stage_x / (1000.0 * pixel_x),
        )
    min_y = min(origin[0] for origin in raw.values())
    min_x = min(origin[1] for origin in raw.values())
    return {
        position: (origin[0] - min_y, origin[1] - min_x)
        for position, origin in raw.items()
    }
