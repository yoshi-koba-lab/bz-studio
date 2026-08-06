"""Generate a synthetic KTL2/.ktf file matching the real Keyence format,
for headless testing when the HDD is disconnected."""
import struct
import io
import numpy as np
from PIL import Image


def encode_keyence_double(val: float) -> int:
    return struct.unpack("<q", struct.pack("<d", val))[0]


def make_ktf(path, width, height, planes=1, calibration_nm=2000.0,
             channel="Channel4", channel_comment="Normal BF", pattern="gradient"):
    tile_size = 512
    import math
    ntx = math.ceil(width / tile_size)
    nty = math.ceil(height / tile_size)
    n_tiles = ntx * nty
    plane_bytes = tile_size * tile_size
    tile_bytes = plane_bytes * planes
    header_size = 112

    # Build a ground-truth full image so tests can compare
    gt = np.zeros((nty * tile_size, ntx * tile_size), dtype=np.uint8)
    yy, xx = np.mgrid[0:nty * tile_size, 0:ntx * tile_size]
    if pattern == "gradient":
        gt = ((xx * 255) // (ntx * tile_size)).astype(np.uint8)
    else:
        gt = (((xx // 32) + (yy // 32)) % 2 * 200 + 30).astype(np.uint8)

    # Data section: store tiles in row-major order
    data = io.BytesIO()
    data.write(b"\x00" * header_size)
    entries = []  # (offset_relative_to_data_section, tile_bytes)
    for ty in range(nty):
        for tx in range(ntx):
            off = data.tell() - header_size  # offset relative to data section start
            tile = gt[ty * tile_size:(ty + 1) * tile_size,
                      tx * tile_size:(tx + 1) * tile_size]
            for _p in range(planes):
                data.write(tile.tobytes())
            entries.append((off, tile_bytes))

    footer_offset = data.tell()

    # Footer: index of (offset, size) uint64 pairs
    for off, sz in entries:
        data.write(struct.pack("<QQ", off, sz))

    # JPEG thumbnail
    thumb = Image.fromarray(gt[:height, :width]).resize((width // 8, height // 8))
    jbuf = io.BytesIO()
    thumb.save(jbuf, format="JPEG")
    data.write(jbuf.getvalue())

    # XML metadata
    cal = encode_keyence_double(calibration_nm)
    na = encode_keyence_double(0.2)
    wd = encode_keyence_double(20.0)
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Data><SingleFileProperty>
<Image><OriginalImageSize><Width Type="System.Int32">{width}</Width><Height Type="System.Int32">{height}</Height></OriginalImageSize>
<Calibration Type="System.Double">{cal}</Calibration><PatchNumber Type="System.Int32">{n_tiles}</PatchNumber></Image>
<Lens><LensName Type="System.String">PlanApo 4x 0.20/20.00mm :Default</LensName><Magnification Type="System.Int32">400</Magnification>
<NumericalAperture Type="System.Double">{na}</NumericalAperture><WorkingDistance Type="System.Double">{wd}</WorkingDistance></Lens>
<Shooting><Channel Type="x">{channel}</Channel><ChannelComment Type="System.String">{channel_comment}</ChannelComment>
<Observation Type="x">Relief</Observation><StageLocationX Type="System.Int32">10460800</StageLocationX>
<StageLocationY Type="System.Int32">82473200</StageLocationY><StageLocationZ Type="System.Int32">3843320</StageLocationZ>
<XyStageRegion><X Type="System.Int32">3446800</X><Y Type="System.Int32">77215200</Y><Width Type="System.Int32">14028000</Width><Height Type="System.Int32">10516000</Height></XyStageRegion>
<Parameter><PixelMode Type="x">Monochrome8Bit</PixelMode><Binnin Type="x">TwoByTwo</Binnin>
<ExposureTime><Numerator Type="System.Int32">600</Numerator><Denominator Type="System.Int32">1000000</Denominator></ExposureTime></Parameter></Shooting>
</SingleFileProperty></Data>"""
    data.write(xml.encode("utf-8"))

    file_size = data.tell()

    # Write header
    buf = bytearray(data.getvalue())
    struct.pack_into("<4s", buf, 0, b"KTL2")
    struct.pack_into("<I", buf, 4, header_size)
    struct.pack_into("<I", buf, 8, header_size)
    struct.pack_into("<Q", buf, 16, footer_offset)
    struct.pack_into("<Q", buf, 24, file_size)
    struct.pack_into("<I", buf, 32, planes)  # image_type ~ planes
    struct.pack_into("<I", buf, 36, tile_size)
    struct.pack_into("<I", buf, 48, width)
    struct.pack_into("<I", buf, 52, height)

    with open(path, "wb") as f:
        f.write(buf)

    return gt[:height, :width]


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Generate synthetic .ktf files for testing.")
    ap.add_argument("-o", "--out", default="synthetic",
                    help="output directory (default: ./synthetic)")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    gt = make_ktf(out / "Test1_A02_CH4.ktf", 7014, 5258, planes=1, pattern="checker")
    np.save(out / "gt_A02_CH4.npy", gt)
    make_ktf(out / "Test1_A02_CH1-2.ktf", 7014, 5258, planes=3, channel="Channel1",
             channel_comment="Alexa 488", pattern="gradient")
    # A second, larger well to test well-switching and auto-downsample
    make_ktf(out / "Test1_B03_CH4.ktf", 9615, 9130, planes=1, pattern="checker")
    make_ktf(out / "Test1_B03_CH1-2.ktf", 9615, 9130, planes=3, channel="Channel1",
             channel_comment="Alexa 488", pattern="gradient")
    print(f"Synthetic files written to {out}")
