=======================================================================
 pSTAM Design Toolbox — asset manifest
=======================================================================
Images and clips shown inside the toolbox. Each has a labelled
placeholder in the interface until the file is present; once a file with
the exact name exists here, it is shown the next time the toolbox starts.

Static images: .png or .jpg.   Animated clips: .gif (they auto-play).

-----------------------------------------------------------------------
 FILE NAME            TAB                 WHAT IT SHOWS
-----------------------------------------------------------------------
 logo.png             Home                Toolbox logo, shown in the masthead.

 gallery_1.png        Home (Gallery)      Gallery carousel images. Add as many
 gallery_2.png                            as you like using this numbering; they
 gallery_3.png                            auto-advance every 3 s and loop, with
 ...                                      left / right arrows and dot markers.
                                          Without gallery_* files the carousel
                                          falls back to hero_collage.png and
                                          pstam_collage.png.

 exploded_view.png    Design Space        Exploded render labelling the four
                                          design variables L, D, T and P.

 pstam_collage.png    Design Space        The fabricated pSTAM units used in
                                          the experiments.

 M1_experiment.gif    Surrogate Models    Static-contraction experiment clip.
 M2_experiment.gif    Surrogate Models    Inflation-time experiment clip.
   (or inflation.gif)                     Either name is accepted.
 M3_experiment.gif    Surrogate Models    Deflation-time experiment clip.
   (or deflation.gif)                     Either name is accepted.

 Clips are scaled to the card width with their aspect ratio preserved and
 shown up to a fixed height. For smooth playback keep each clip near
 1120 x 960 px or smaller, 12-15 fps and a few MB.

-----------------------------------------------------------------------
 readme/   Media for the GitHub page only (not used by the toolbox).
           See readme/README.md.
-----------------------------------------------------------------------
