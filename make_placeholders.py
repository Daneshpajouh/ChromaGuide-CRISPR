import zlib
import struct

def make_png(width, height):
    # RGB 1x1 pixel
    width_bytes = width.to_bytes(4, 'big')
    height_bytes = height.to_bytes(4, 'big')

    # 8-bit depth, RGB (2), default compression/filter/interlace
    ihdr_data = width_bytes + height_bytes + b'\x08\x02\x00\x00\x00'
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data).to_bytes(4, 'big')
    ihdr = b'\x00\x00\x00\x0dIHDR' + ihdr_data + ihdr_crc

    # Data: 1 pixel (light gray)
    # scanline: 0 (filter type) + R(200) + G(200) + B(200)
    raw_data = b'\x00' + b'\xc8\xc8\xc8' * width * height # Simple gray
    compressed = zlib.compress(raw_data)
    idat_len = len(compressed).to_bytes(4, 'big')
    idat_crc = zlib.crc32(b'IDAT' + compressed).to_bytes(4, 'big')
    idat = idat_len + b'IDAT' + compressed + idat_crc

    # IEND
    iend = b'\x00\x00\x00\x00IEND\xae\x42\x60\x82'

    return b'\x89PNG\r\n\x1a\n' + ihdr + idat + iend

files = [
    'figures/fig_mechanism.png',
    'figures/fig_complexity_graph.png',
    'figures/fig_architecture.png',
    'figures/fig_epigenomics.png',
    'figures/fig_conformal.png'
]

data = make_png(400, 300) # 400x300 placeholder
for f in files:
    with open(f, 'wb') as out:
        out.write(data)
    print(f"Created {f}")
