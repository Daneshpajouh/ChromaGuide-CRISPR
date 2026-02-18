import zlib

def make_png(width, height, color_val=200):
    # RGB 1x1 pixel (scaled)
    width_bytes = width.to_bytes(4, 'big')
    height_bytes = height.to_bytes(4, 'big')
    ihdr_data = width_bytes + height_bytes + b'\x08\x02\x00\x00\x00'
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data).to_bytes(4, 'big')
    ihdr = b'\x00\x00\x00\x0dIHDR' + ihdr_data + ihdr_crc

    line_data = b'\x00' + (bytes([color_val, color_val, color_val]) * width)
    raw_data = line_data * height
    compressed = zlib.compress(raw_data)

    idat_len = len(compressed).to_bytes(4, 'big')
    idat_crc = zlib.crc32(b'IDAT' + compressed).to_bytes(4, 'big')
    idat = idat_len + b'IDAT' + compressed + idat_crc
    iend = b'\x00\x00\x00\x00IEND\xae\x42\x60\x82'
    return b'\x89PNG\r\n\x1a\n' + ihdr + idat + iend

figures = [
    "fig_1_1.png", "fig_1_2.png", "fig_1_3.png", "fig_1_4.png", "fig_1_5.png", "fig_1_6.png",
    "fig_2_1.png", "fig_2_2.png", "fig_2_3.png", "fig_2_4.png", "fig_2_5.png",
    "fig_3_1.png", "fig_3_2.png", "fig_3_3.png", "fig_3_4.png",
    "fig_4_1.png", "fig_4_2.png", "fig_4_3.png", "fig_4_4.png", "fig_4_5.png", "fig_4_6.png",
    "fig_5_1.png", "fig_5_2.png", "fig_5_3.png", "fig_5_4.png",
    "fig_6_1.png", "fig_6_2.png", "fig_6_3.png", "fig_6_4.png", "fig_6_5.png",
    "fig_7_1.png", "fig_7_2.png", "fig_7_3.png", "fig_7_4.png",
    "fig_8_1.png", "fig_8_2.png", "fig_8_3.png", "fig_8_4.png",
    "fig_9_1.png", "fig_9_2.png", "fig_9_3.png", "fig_9_4.png", "fig_9_5.png",
    "fig_10_1.png", "fig_10_2.png", "fig_10_3.png", "fig_10_4.png",
    "fig_11_1.png", "fig_11_2.png", "fig_11_3.png",
    "fig_12_1.png", "fig_12_2.png", "fig_12_3.png", "fig_12_4.png"
]

for f in figures:
    data = make_png(800, 400, color_val=220) # Light gray 800x400
    with open(f"figures/{f}", 'wb') as out:
        out.write(data)
    print(f"Created figures/{f}")
