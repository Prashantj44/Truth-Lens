import shutil
from pathlib import Path
from PIL import Image, ImageDraw

# 1. Create SVG
svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="22" ry="22" fill="#ffffff"/>
  <g transform="translate(14, 14) scale(3)">
    <path d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" 
          fill="none" 
          stroke="#000000" 
          stroke-width="1.8" 
          stroke-linecap="round" 
          stroke-linejoin="round"/>
  </g>
</svg>"""

Path("favicon.svg").write_text(svg_content, encoding="utf-8")

# 2. Create crisp multi-size ICO & PNG
sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img_256 = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(img_256)

# White squircle
draw.rounded_rectangle([8, 8, 248, 248], radius=56, fill=(255, 255, 255, 255))

# Cube vertices mapped to 256x256
scale = 7.0
cx, cy = 128, 128

def t(x, y):
    return (cx + (x - 12) * scale, cy + (y - 12) * scale)

pt_top = t(12, 3)
pt_tr  = t(20, 7)
pt_tl  = t(4, 7)
pt_c   = t(12, 11)
pt_br  = t(20, 17)
pt_bl  = t(4, 17)
pt_b   = t(12, 21)

w = 14
# Outline hexagon
draw.line([pt_top, pt_tr], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_tr, pt_br], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_br, pt_b], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_b, pt_bl], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_bl, pt_tl], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_tl, pt_top], fill=(0, 0, 0, 255), width=w, joint="curve")

# Inner 3 edges meeting at center
draw.line([pt_tl, pt_c], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_tr, pt_c], fill=(0, 0, 0, 255), width=w, joint="curve")
draw.line([pt_b, pt_c], fill=(0, 0, 0, 255), width=w, joint="curve")

icon_images = [img_256.resize(s, Image.Resampling.LANCZOS) for s in sizes[:-1]]
img_256.save("favicon.ico", format="ICO", sizes=sizes, append_images=icon_images)
img_256.save("favicon.png", format="PNG")
img_256.save("apple-touch-icon.png", format="PNG")

# 3. Copy to all required destinations: public, frontend, root
for d in [Path("public"), Path("frontend")]:
    d.mkdir(parents=True, exist_ok=True)
    for name in ["favicon.ico", "favicon.svg", "favicon.png", "apple-touch-icon.png"]:
        shutil.copy(name, d / name)
        print(f"Copied {name} -> {d}")

print("All favicon assets generated and synchronized successfully.")
