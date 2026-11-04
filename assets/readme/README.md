# Media for the repository page

These files appear only on the GitHub page (`README.md` at the repository root); the toolbox itself does not read this folder. To update any of them, replace the file under exactly the same name, then commit and push.

| File | Where it appears | Format |
|:--|:--|:--|
| `toolbox_walkthrough.webp` | Top of the page, under the contact badges | Animated WebP, 1280 × 720, looping |
| `tab_1_home.png` | *The five tabs*, row 1 left | Screenshot of the Home tab |
| `tab_2_design_space.png` | *The five tabs*, row 1 right | Screenshot of the Design Space tab |
| `tab_3_surrogate_models.png` | *The five tabs*, row 2 left | Screenshot of the Surrogate Models tab |
| `tab_4_objective_cost.png` | *The five tabs*, row 2 right | Screenshot of the Objective & Cost tab |
| `tab_5_optimise_report.png` | *The five tabs*, row 3, centred | Screenshot of the Optimise & Report tab |

Take all five screenshots at the same window size so the grid lines up.

## Re-making the walkthrough clip

Record the screen while you go from the Home tab to the exported report, save it as an MP4, then convert it with [ffmpeg](https://ffmpeg.org/):

```bash
ffmpeg -i walkthrough.mp4 -vf "fps=12,scale=1280:-1:flags=lanczos" \
       -c:v libwebp -lossless 0 -quality 70 -compression_level 6 \
       -loop 0 -an assets/readme/toolbox_walkthrough.webp
```

- `-loop 0` makes it loop forever; without it the clip plays once and stops.
- If the result exceeds ~10 MB, lower `-quality` (for example 55), lower `fps` to 10, or trim the recording with `-ss <start> -t <duration>` before `-i`.
- An animated WebP is used rather than a video because GitHub autoplays and loops it inside an `<img>` tag; GitHub strips `autoplay` and `loop` from `<video>` tags.
