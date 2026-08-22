#!/usr/bin/env python3
"""Inline the real photographs into the mock and write the publishable file.

The mock is kept as a template with {{name}} holes rather than a finished file
because a base64 JPEG is thousands of characters on one line: with the images
inlined, the source becomes unreadable and uneditable, and every edit risks
corrupting an image by a stray character.
"""
import io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
imgs = json.load(open(os.path.join(HERE, "images.json")))
src = io.open(os.path.join(HERE, "template.html"), encoding="utf-8").read()

holes = set(re.findall(r"\{\{(\w+)\}\}", src))
missing = sorted(holes - set(imgs))
if missing:
    sys.exit(f"template references images that do not exist: {missing}")
unused = sorted(set(imgs) - holes)
if unused:
    print(f"  note: prepared but unused: {unused}")

out = re.sub(r"\{\{(\w+)\}\}", lambda m: imgs[m.group(1)], src)

# A hole left behind is a broken image the eye skips over in a long page.
assert "{{" not in out, "unsubstituted hole remains"

dst = os.path.join(HERE, "lookzi-product-book.html")
io.open(dst, "w", encoding="utf-8").write(out)
print(f"  {len(holes)} images inlined -> {dst}")
print(f"  {len(out) / 1024:.0f} KB")
