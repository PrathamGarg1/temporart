#!/usr/bin/env python3
import base64
from pathlib import Path

root = Path(__file__).resolve().parent
assets = root / "html-slides-assets"
img = root / "img"


def uri(name: str) -> str:
    p = img / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


css = (assets / "suraksha-overrides.css").read_text()
body = (
    (assets / "slides-body.html")
    .read_text()
    .replace("{{IMG_WA}}", uri("wa-flag-mock.png"))
    .replace("{{IMG_WAHI}}", uri("wa-devanagari.png"))
    .replace("{{IMG_DESK}}", uri("on-device-desk-light.png"))
    .replace("{{IMG_PHONE}}", uri("macd-phone-light.png"))
)
scene = (assets / "suraksha-scene.js").read_text()
motion = (assets / "suraksha-motion.js").read_text()
runtime = (assets / "slides-runtime.js").read_text()
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SurakshaNet · B.Tech project defence</title>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.7/dist/gsap.min.js"></script>
  <style>
{css}
  </style>
</head>
<body>
{body}
<script>
{scene}
</script>
<script>
{motion}
</script>
<script>
{runtime}
</script>
</body>
</html>
"""
out = root / "btech.html"
out.write_text(html)
(root / "apple.html").write_text(html)
(root.parent / "codemix_eval" / "SurakshaNet_Guide.html").write_text(html)
print(out, "bytes", out.stat().st_size)
