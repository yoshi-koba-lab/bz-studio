"""
KTL2/BZ03 .ktf file reader — memory-efficient version.

Uses mmap for file access and supports downsampled reading to handle
files up to 500MB+ without excessive memory usage.
"""

from __future__ import annotations

import struct
import math
import mmap
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def decode_packed_double(raw: str) -> float:
    """The format stores System.Double values as their int64 bit pattern in XML.

    e.g. calibration "4656510908468559872" -> 2000.0 (nm/pixel).
    """
    try:
        return struct.unpack("<d", struct.pack("<q", int(raw)))[0]
    except (ValueError, struct.error):
        return 0.0


@dataclass
class KtfMetadata:
    width: int = 0
    height: int = 0
    tile_size: int = 512
    image_type: int = 0  # 1=fluorescence(3-plane), 2=brightfield(1-plane)
    channel: str = ""
    channel_comment: str = ""
    observation_mode: str = ""
    lens_name: str = ""
    magnification: int = 0
    pixel_mode: str = ""
    exposure_numerator: int = 0
    exposure_denominator: int = 1
    stage_x: int = 0
    stage_y: int = 0
    stage_z: int = 0
    patch_count: int = 0
    binning: str = ""
    calibration_nm: float = 0.0  # nanometers per pixel
    numerical_aperture: float = 0.0
    working_distance_mm: float = 0.0
    region_x_nm: int = 0  # stage-region top-left X (nm)
    region_y_nm: int = 0  # stage-region top-left Y (nm)
    raw_xml: str = ""

    @property
    def um_per_pixel(self) -> float:
        return self.calibration_nm / 1000.0


@dataclass
class KtfInfo:
    """Lightweight info extracted from header/footer only (no pixel data loaded)."""
    path: Path
    metadata: KtfMetadata
    header_size: int
    footer_offset: int
    file_size: int
    tile_entry_count: int
    tile_byte_size: int  # bytes per tile entry
    thumbnail_jpeg: Optional[bytes] = None

    @property
    def well_id(self) -> str:
        name = self.path.stem
        for part in name.split("_"):
            if len(part) >= 2 and part[0].isalpha() and part[1:].isdigit():
                return part
        return ""

    @property
    def channel_id(self) -> str:
        name = self.path.stem
        for part in name.split("_"):
            if part.startswith("CH"):
                return part
        return ""


class KtfFormatError(ValueError):
    """Raised when a file is not a readable KTL2/.ktf image."""


def _read_header(f) -> tuple:
    f.seek(0)
    h = f.read(112)
    if len(h) < 112:
        raise KtfFormatError(f"Truncated KTF header ({len(h)} bytes, need 112)")
    magic = h[0:4]
    if magic != b"KTL2":
        raise KtfFormatError(f"Not a KTL2 file: magic={magic!r}")
    header_size = struct.unpack("<I", h[4:8])[0]
    footer_offset = struct.unpack("<Q", h[16:24])[0]
    file_size = struct.unpack("<Q", h[24:32])[0]
    image_type = struct.unpack("<I", h[32:36])[0]
    tile_size = struct.unpack("<I", h[36:40])[0]
    width = struct.unpack("<I", h[48:52])[0]
    height = struct.unpack("<I", h[52:56])[0]
    return header_size, footer_offset, file_size, image_type, tile_size, width, height


def _read_tile_index_info(f, footer_offset: int, file_size: int) -> tuple:
    """Read just enough of the tile index to get count and tile byte size."""
    f.seek(footer_offset)
    # Read first entry to get tile_byte_size
    chunk = f.read(16)
    if len(chunk) < 16:
        return 0, 0, []
    first_off, first_sz = struct.unpack("<QQ", chunk)
    if first_sz == 0:
        return 0, 0, []

    tile_byte_size = first_sz
    # Read remaining entries
    entries = [(first_off, first_sz)]
    max_read = min(file_size - footer_offset, 100000 * 16)  # safety limit
    remaining = f.read(max_read - 16)
    pos = 0
    while pos + 16 <= len(remaining):
        off, sz = struct.unpack("<QQ", remaining[pos:pos + 16])
        if sz != tile_byte_size:
            break
        entries.append((off, sz))
        pos += 16

    return len(entries), tile_byte_size, entries


def _extract_thumbnail_from_file(f, footer_offset: int, num_entries: int) -> Optional[bytes]:
    """Read JPEG thumbnail from footer area without loading full file."""
    pos = footer_offset + num_entries * 16
    f.seek(pos)
    # Read a reasonable chunk (thumbnails are typically <500KB)
    chunk = f.read(512 * 1024)
    jpeg_start = chunk.find(b"\xff\xd8\xff")
    if jpeg_start < 0:
        return None
    jpeg_end = chunk.find(b"\xff\xd9", jpeg_start + 3)
    if jpeg_end < 0:
        return None
    return bytes(chunk[jpeg_start:jpeg_end + 2])


def _parse_xml_from_file(f, footer_offset: int, num_entries: int) -> KtfMetadata:
    """Read XML metadata from footer area."""
    meta = KtfMetadata()
    pos = footer_offset + num_entries * 16
    f.seek(pos)
    # Read enough for thumbnail + XML (usually < 1MB total)
    chunk = f.read(1024 * 1024)
    xml_start = chunk.find(b"<?xml")
    if xml_start < 0:
        return meta
    xml_end = chunk.find(b"</Data>", xml_start)
    if xml_end < 0:
        return meta
    xml_bytes = chunk[xml_start:xml_end + 7]
    meta.raw_xml = xml_bytes.decode("utf-8-sig", errors="replace")
    try:
        root = ET.fromstring(meta.raw_xml)
        sfp = root.find("SingleFileProperty")
        if sfp is None:
            return meta
        img = sfp.find("Image")
        if img is not None:
            w = img.find("OriginalImageSize/Width")
            h = img.find("OriginalImageSize/Height")
            if w is not None:
                meta.width = int(w.text)
            if h is not None:
                meta.height = int(h.text)
            pc = img.find("PatchNumber")
            if pc is not None:
                meta.patch_count = int(pc.text)
            cal = img.find("Calibration")
            if cal is not None and cal.text:
                meta.calibration_nm = decode_packed_double(cal.text)
        lens = sfp.find("Lens")
        if lens is not None:
            ln = lens.find("LensName")
            if ln is not None:
                meta.lens_name = ln.text or ""
            mg = lens.find("Magnification")
            if mg is not None:
                meta.magnification = int(mg.text)
            na = lens.find("NumericalAperture")
            if na is not None and na.text:
                meta.numerical_aperture = decode_packed_double(na.text)
            wd = lens.find("WorkingDistance")
            if wd is not None and wd.text:
                meta.working_distance_mm = decode_packed_double(wd.text)
        shoot = sfp.find("Shooting")
        if shoot is not None:
            for tag, attr in [("Channel", "channel"), ("ChannelComment", "channel_comment"),
                              ("Observation", "observation_mode")]:
                el = shoot.find(tag)
                if el is not None:
                    setattr(meta, attr, el.text or "")
            for tag, attr in [("StageLocationX", "stage_x"), ("StageLocationY", "stage_y"),
                              ("StageLocationZ", "stage_z")]:
                el = shoot.find(tag)
                if el is not None:
                    setattr(meta, attr, int(el.text))
            region = shoot.find("XyStageRegion")
            if region is not None:
                rx = region.find("X")
                ry = region.find("Y")
                if rx is not None and rx.text:
                    meta.region_x_nm = int(rx.text)
                if ry is not None and ry.text:
                    meta.region_y_nm = int(ry.text)
            param = shoot.find("Parameter")
            if param is not None:
                pm = param.find("PixelMode")
                if pm is not None:
                    meta.pixel_mode = pm.text or ""
                bn = param.find("Binnin")
                if bn is not None:
                    meta.binning = bn.text or ""
                et_el = param.find("ExposureTime")
                if et_el is not None:
                    num = et_el.find("Numerator")
                    den = et_el.find("Denominator")
                    if num is not None:
                        meta.exposure_numerator = int(num.text)
                    if den is not None:
                        meta.exposure_denominator = int(den.text)
    except ET.ParseError:
        pass
    return meta


def scan_ktf_light(path: Path) -> KtfInfo:
    """Lightweight scan: reads only header + footer (no pixel data). ~1ms per file."""
    with open(path, "rb") as f:
        header_size, footer_offset, file_size, image_type, tile_size, width, height = _read_header(f)
        num_entries, tile_byte_size, _ = _read_tile_index_info(f, footer_offset, file_size)
        thumbnail = _extract_thumbnail_from_file(f, footer_offset, num_entries)
        metadata = _parse_xml_from_file(f, footer_offset, num_entries)
        metadata.width = width
        metadata.height = height
        metadata.tile_size = tile_size
        metadata.image_type = image_type

    return KtfInfo(
        path=path,
        metadata=metadata,
        header_size=header_size,
        footer_offset=footer_offset,
        file_size=file_size,
        tile_entry_count=num_entries,
        tile_byte_size=tile_byte_size,
        thumbnail_jpeg=thumbnail,
    )


def snap_downsample(ds: int, tile_size: int = 512) -> int:
    """Largest power-of-two <= ds that still divides tile_size.

    The mosaic is decimated tile-by-tile, so the sampling grid only stays
    continuous across tile borders when ds divides tile_size. A factor like 3 or
    10 would restart the phase at every tile — shifting each tile's content and
    dropping pixels at the right/bottom edges. Snapping keeps the geometry exact;
    callers that need a precise output size should resize afterwards.
    """
    ds = max(1, int(ds))
    snapped = 1
    while snapped * 2 <= ds and tile_size % (snapped * 2) == 0:
        snapped *= 2
    return snapped


def reconstruct_image(path: Path, downsample: int = 1, plane: int = 0,
                      progress_callback=None) -> np.ndarray:
    """Reconstruct stitched image using mmap (memory-efficient).

    Args:
        path: Path to .ktf file
        downsample: Requested decimation. Snapped down to a power of two so the
            tile grid stays aligned (see snap_downsample); the returned array may
            therefore be larger than width/downsample.
        plane: Which plane to read (0=primary) for multi-plane tiles
        progress_callback: Optional callable(current, total) for progress updates
    """
    with open(path, "rb") as f:
        header_size, footer_offset, file_size, image_type, tile_size, width, height = _read_header(f)
        num_entries, tile_byte_size, entries = _read_tile_index_info(f, footer_offset, file_size)

    ds = snap_downsample(downsample, tile_size)

    if not entries:
        return np.zeros((-(-height // ds), -(-width // ds)), dtype=np.uint8)

    ntx = math.ceil(width / tile_size)
    nty = math.ceil(height / tile_size)
    n_image_tiles = ntx * nty
    plane_bytes = tile_size * tile_size

    # ds divides tile_size exactly, so tiles tessellate without phase drift.
    out_tile = tile_size // ds
    out_h = nty * out_tile
    out_w = ntx * out_tile
    image = np.zeros((out_h, out_w), dtype=np.uint8)

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for idx in range(min(n_image_tiles, len(entries))):
                if progress_callback and idx % 50 == 0:
                    progress_callback(idx, n_image_tiles)

                ty = idx // ntx
                tx = idx % ntx
                file_offset = entries[idx][0] + header_size
                plane_offset = file_offset + plane * plane_bytes

                if plane_offset + plane_bytes > len(mm):
                    continue

                tile_raw = mm[plane_offset:plane_offset + plane_bytes]
                tile = np.frombuffer(tile_raw, dtype=np.uint8).reshape(tile_size, tile_size)

                if ds > 1:
                    tile = tile[::ds, ::ds]

                y0 = ty * out_tile
                x0 = tx * out_tile
                th, tw = tile.shape
                # clip to the destination slot (defensive; exact when ds | tile_size)
                sh = min(th, out_h - y0)
                sw = min(tw, out_w - x0)
                image[y0:y0 + sh, x0:x0 + sw] = tile[:sh, :sw]
        finally:
            mm.close()

    # Crop to actual image size (ceil to match the downsampled grid)
    final_h = -(-height // ds)
    final_w = -(-width // ds)
    return image[:final_h, :final_w]


def reconstruct_image_full(path: Path, plane: int = 0,
                           progress_callback=None) -> np.ndarray:
    """Full-resolution reconstruction using mmap."""
    return reconstruct_image(path, downsample=1, plane=plane,
                             progress_callback=progress_callback)


def reconstruct_region(path: Path, x0: int, y0: int, x1: int, y1: int,
                       downsample: int = 1, plane: int = 0) -> np.ndarray:
    """Reconstruct only a rectangular region [x0,y0)-[x1,y1) in full-res pixel coords.

    Loads just the tiles that intersect the region, so it stays fast and
    low-memory even for 500MB files. Used for detail-on-zoom.
    """
    with open(path, "rb") as f:
        header_size, footer_offset, file_size, image_type, tile_size, width, height = _read_header(f)
        num_entries, tile_byte_size, entries = _read_tile_index_info(f, footer_offset, file_size)

    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))

    if not entries:
        return np.zeros(((y1 - y0) // downsample, (x1 - x0) // downsample), dtype=np.uint8)

    ntx = math.ceil(width / tile_size)
    plane_bytes = tile_size * tile_size

    tcx0, tcx1 = x0 // tile_size, (x1 - 1) // tile_size
    tcy0, tcy1 = y0 // tile_size, (y1 - 1) // tile_size

    buf_w = (tcx1 - tcx0 + 1) * tile_size
    buf_h = (tcy1 - tcy0 + 1) * tile_size
    buf = np.zeros((buf_h, buf_w), dtype=np.uint8)

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for ty in range(tcy0, tcy1 + 1):
                for tx in range(tcx0, tcx1 + 1):
                    idx = ty * ntx + tx
                    if idx >= len(entries):
                        continue
                    plane_offset = entries[idx][0] + header_size + plane * plane_bytes
                    if plane_offset + plane_bytes > len(mm):
                        continue
                    tile = np.frombuffer(
                        mm[plane_offset:plane_offset + plane_bytes], dtype=np.uint8
                    ).reshape(tile_size, tile_size)
                    by = (ty - tcy0) * tile_size
                    bx = (tx - tcx0) * tile_size
                    buf[by:by + tile_size, bx:bx + tile_size] = tile
        finally:
            mm.close()

    # Crop the tile-aligned buffer to the exact requested region
    cy0 = y0 - tcy0 * tile_size
    cx0 = x0 - tcx0 * tile_size
    crop = buf[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)]
    if downsample > 1:
        crop = crop[::downsample, ::downsample]
    return np.ascontiguousarray(crop)


def is_ktf_file(p: Path) -> bool:
    """True for real .ktf data files (case-insensitive), excluding macOS sidecars.

    External drives are usually exFAT/NTFS, where macOS writes AppleDouble
    companions named `._Foo.ktf`. Those match the extension but are not images.
    """
    return p.suffix.lower() == ".ktf" and not p.name.startswith("._")


def scan_experiment_folder(folder: Path) -> dict:
    """Scan an experiment folder and return organized structure.

    Unreadable/corrupt .ktf files are skipped and reported in result["errors"]
    rather than aborting the whole scan.
    """
    folder = Path(folder)
    result = {
        "name": folder.name,
        "path": folder,
        "wells": {},
        "ici_path": None,
        "ibc2_path": None,
        "errors": [],
    }

    for f in sorted(folder.iterdir()):
        suffix = f.suffix.lower()
        if is_ktf_file(f):
            try:
                info = scan_ktf_light(f)
            except Exception as e:  # corrupt, truncated, still copying, unreadable…
                result["errors"].append((f.name, str(e)))
                continue
            well = info.well_id
            ch = info.channel_id
            if well and ch:
                if well not in result["wells"]:
                    result["wells"][well] = {}
                result["wells"][well][ch] = info
        elif suffix == ".ici":
            result["ici_path"] = f
        elif suffix == ".ibc2":
            result["ibc2_path"] = f

    return result
