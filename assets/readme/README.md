# Media for the repository page

These files appear only on the GitHub page (`README.md` at the repository root); the toolbox itself does not read this folder. The page loads small, web-optimised copies so it opens quickly; the full-size originals stay in the repository and open when a visitor clicks a screenshot.

| Page element | Loaded by the page | Full-size original |
|:--|:--|:--|
| Logo | `logo.webp` (600 px) | `../logo.png` |
| Walkthrough clip | `toolbox_walkthrough.webp` (1024 × 576, looping) | Not kept separately |
| *How it works* diagram | `how_it_works.png` | `source/how_it_works.html` |
| *The five tabs* | `thumbs/tab_*.webp` (1100 px) | `tab_1_home.png` … `tab_5_optimise_report.png` |

## Updating the media

1. Replace the full-size file under the same name: `assets/logo.png`, a `tab_*.png` screenshot, or `toolbox_walkthrough.webp`.
2. From the repository root, regenerate the web copies:

   ```bash
   pip install pillow
   python assets/readme/source/build_readme_media.py
   ```

   This rewrites `logo.webp` and `thumbs/`, and shrinks the walkthrough clip to 1024 px wide if it is larger, keeping its timing and infinite loop.
3. Commit and push.

Take all five screenshots at the same window size so the grid lines up.

## Editing the diagram

The diagram is drawn from `source/how_it_works.html`. Edit the labels there, open the file in Chrome or Edge, and capture the card at 2× zoom (DevTools → *Capture node screenshot* on the `#card` element). Save the result as `how_it_works.png`. It is an image rather than a Mermaid block because GitHub renders Mermaid slowly and sometimes fails to render it at all.

## Re-making the walkthrough clip

Record the screen while you go from the Home tab to the exported report, save it as an MP4, then convert it with [ffmpeg](https://ffmpeg.org/):

```bash
ffmpeg -i walkthrough.mp4 -vf "fps=12,scale=1024:-1:flags=lanczos" \
       -c:v libwebp -lossless 0 -quality 70 -compression_level 6 \
       -loop 0 -an assets/readme/toolbox_walkthrough.webp
```

- `-loop 0` makes it loop forever; without it the clip plays once and stops.
- Keep the result under about 1 MB for a fast page: lower `-quality` (for example 55), lower `fps` to 10, or trim the recording with `-ss <start> -t <duration>` before `-i`.
- An animated WebP is used rather than a video because GitHub autoplays and loops it inside an `<img>` tag; GitHub strips `autoplay` and `loop` from `<video>` tags.
