These four precomputed examples use public WTBD source images and the app's
frozen detector, crop preprocessing, classifier, and Grad-CAM implementation.
`cases.json` orders the carousel; each case contains matching image previews
and the real classifier scores in `prediction.json`. Regenerate them with
`python scripts/build_landing_examples.py` from the repository root.

The carousel covers four different classifier outputs: Craze (image 1),
Surface injury (image 31), Thunderstrike (image 10), and Corrosion (image 3).

Source: *Multiclass Dataset for Intelligent Detection of Wind Turbine Blade
Defects Using Drone Imagery*, Lipeng Ji, Junjie Cheng, and Shilong Wu,
[WTBD version 1](https://doi.org/10.6084/m9.figshare.30210175.v1), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source previews are re-encoded; region previews add detector boxes;
crop previews apply contextual cropping and resizing; attention previews add
Grad-CAM overlays. No private uploads or location metadata are included.
