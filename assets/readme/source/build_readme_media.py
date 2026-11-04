"""
Rebuild the lightweight media shown on the GitHub page.

Run from the repository root after replacing any full-size file:
    pip install pillow
    python assets/readme/source/build_readme_media.py

Inputs (full size, kept in the repository and opened on click):
    assets/logo.png
    assets/readme/tab_1_home.png ... tab_5_optimise_report.png
    assets/readme/toolbox_walkthrough.webp   (re-encoded in place if wider than 1024 px)
Outputs (what the page loads):
    assets/readme/logo.webp
    assets/readme/thumbs/tab_*.webp
"""
import os
import struct
from PIL import Image, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MEDIA = os.path.join(ROOT, "assets", "readme")
TABS = ["tab_1_home", "tab_2_design_space", "tab_3_surrogate_models",
        "tab_4_objective_cost", "tab_5_optimise_report"]


def resized(im, width):
    if im.width <= width:
        return im
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def webp_durations(path):
    data = open(path, "rb").read()
    i, out = 12, []
    while i < len(data):
        tag = data[i:i + 4]
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        if tag == b"ANMF":
            out.append(int.from_bytes(data[i + 8 + 12:i + 8 + 15], "little"))
        i += 8 + size + (size & 1)
    return out


def report(path):
    print(f"  {os.path.relpath(path, ROOT):48s} {os.path.getsize(path) / 1024:7.0f} KB")


def main():
    # Logo: shown at 300 px, stored at 600 px for high-density screens.
    logo = resized(Image.open(os.path.join(ROOT, "assets", "logo.png")).convert("RGB"), 600)
    out = os.path.join(MEDIA, "logo.webp")
    logo.save(out, "WEBP", quality=88, method=6)
    report(out)

    # Tab screenshots: 1100 px thumbnails; the full PNG opens on click.
    os.makedirs(os.path.join(MEDIA, "thumbs"), exist_ok=True)
    for name in TABS:
        src = os.path.join(MEDIA, name + ".png")
        if not os.path.exists(src):
            print(f"  missing {src}"); continue
        out = os.path.join(MEDIA, "thumbs", name + ".webp")
        resized(Image.open(src).convert("RGB"), 1100).save(out, "WEBP", quality=88, method=6)
        report(out)

    # Walkthrough clip: keep it at most 1024 px wide, looping forever.
    clip = os.path.join(MEDIA, "toolbox_walkthrough.webp")
    im = Image.open(clip)
    if getattr(im, "n_frames", 1) > 1 and im.width > 1024:
        durations = webp_durations(clip)
        frames = [resized(f.convert("RGB"), 1024) for f in ImageSequence.Iterator(im)]
        im.close()
        frames[0].save(clip, "WEBP", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, quality=70, method=4, minimize_size=True)
    report(clip)


if __name__ == "__main__":
    main()
