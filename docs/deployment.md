# BladeScope Streamlit Community Cloud deployment

Application v3 is publicly deployed as a Streamlit research demonstration at [https://bladescope.streamlit.app/](https://bladescope.streamlit.app/). This public demo is not evidence of production, safety, or external-domain readiness.

## Deployment coordinates

- Repository: `Stukhori/BadeScope`
- Branch: `main`
- Main file path: `app/app.py`
- Public URL: [https://bladescope.streamlit.app/](https://bladescope.streamlit.app/)
- Python: `3.11`
- Dependency declaration: `app/requirements.txt`
- System-package declaration: none
- Streamlit configuration: `.streamlit/config.toml`
- Secrets: none

The public app hides Community Cloud's creator-avatar profile link with a
same-origin stylesheet injected by `windblade_demo.chrome`. Streamlit's hosting
badge remains visible. This is a presentation preference, not access control:
the public creator profile still exists, and changes to Community Cloud's wrapper
markup may require updating the selector. Cross-origin embedding is left alone.

The entrypoint-local dependency file pins Streamlit `1.62.0`, streamlit-cropper `0.3.1`, OpenCV Headless `4.11.0.86`, Ultralytics `8.3.150`, PyTorch `2.13.0+cpu`, torchvision `0.28.0+cpu`, and the remaining validated application packages. The exact frozen crop-classifier checkpoint and detector proposal checkpoint are tracked. No runtime model download is required.

The application does not call OpenCV GUI, Qt/GTK window, or OpenGL display functions. It therefore uses only `opencv-python-headless==4.11.0.86`. The repository deliberately has no root `packages.txt`, so Streamlit Community Cloud skips apt processing and the application does not depend on `libGL.so.1` or `libgthread-2.0.so.0`.

Ultralytics 8.3.150 normally declares `opencv-python>=4.6.0`; pip does not treat the headless distribution as satisfying that differently named requirement and would install both distributions. The tracked deployment wheel is built reproducibly from the official Ultralytics 8.3.150 source, retains an identical Python payload, and changes only that dependency metadata to the exact headless pin. Its provenance and hashes are recorded in `app/vendor/README.md`.

## Deploy

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with a GitHub account that can access the repository.
2. Select **Create app** and choose the existing repository option.
3. Use branch `main` and enter `app/app.py` as the main file path.
4. In **Advanced settings**, select Python 3.11. No secrets are required.
5. Deploy. Streamlit runs from the repository root and reads `.streamlit/config.toml`.

## Validate locally

```powershell
uv run python scripts/validate_deployment.py
uv run streamlit run app/app.py --server.address 127.0.0.1
```

The validator requires both checkpoints, the vendored Ultralytics wheel, and all deployment inputs to be tracked. It requires `packages.txt` to be absent, verifies the wheel identity and headless-only OpenCV metadata, verifies checkpoint byte identities, loads the frozen classifier on CPU, and checks the pinned dependency and Streamlit configuration contract.

## Scope

The experimental detector runs on CPU with fixed image size 640, threshold `0.39`, NMS IoU `0.7`, class-agnostic NMS, and maximum 300 detections. Users cannot change detector controls and must review proposals before classification. Prepared-crop and manual-region workflows remain available.

Uploads, proposals, crops, session history, visualizations, and exports remain in process memory. The app disables telemetry and external trackers, performs no external API call or runtime artifact download, and writes no prediction output. The public deployment does not alter scientific results and does not establish production readiness, safety fitness, or performance on external operational imagery.
