"""Static, source-grounded introduction; no inference runs when Home opens."""
from base64 import b64encode
from html import escape
import json
from pathlib import Path

import streamlit as st

from windblade_demo.constants import HUMAN_LABELS


def render_landing(root: Path) -> None:
    base = root / "app/assets/landing"
    cases = json.loads((base / "cases.json").read_text(encoding="utf-8"))
    index = st.session_state.get("landing_case", 0) % len(cases)

    def change_case(step: int) -> None:
        st.session_state["landing_case"] = (index + step) % len(cases)

    assets = base / cases[index]["directory"]
    prediction = json.loads((assets / "prediction.json").read_text(encoding="utf-8"))

    def image(name: str, alt: str) -> str:
        mime = "image/jpeg" if name.endswith(".jpg") else "image/png"
        data = b64encode((assets / name).read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}" alt="{escape(alt)}">'

    bars = "".join(
        '<div class="preview-score"><div>'
        f'<span>{escape(HUMAN_LABELS[label])}</span><strong>{score:.1%}</strong></div>'
        f'<div class="preview-track"><span style="width:{score * 100:.3f}%"></span></div></div>'
        for label, score in sorted(prediction["scores"].items(), key=lambda item: item[1], reverse=True)
    )
    categories = "".join(f'<span>{escape(label)}</span>' for label in HUMAN_LABELS.values())
    introduction = (
        '<section class="landing landing-intro">'
        '<div class="eyebrow">BLADESCOPE / BLADE DEFECT RECOGNITION</div>'
        '<h1>See the damage.<br>Understand the prediction.</h1>'
        '<p>Turn blade inspection images into regions you can review, predictions you can compare, '
        'and visual explanations you can explore.</p>'
        '<a class="classify-link" href="#classification-workspace" target="_self">Classify <span aria-hidden="true">↓</span></a>'
        '<div class="landing-meta">6 defect categories · Region review · Local processing</div></section>'
    )
    preview = (
        '<div class="preview-toolbar"><span class="preview-dot"></span>'
        f'DETECTION PREVIEW <span>CASE {index + 1:02d} / REGION PROPOSALS</span></div>'
    )
    html = (
        '<section class="landing">'
        '<div class="landing-section-label"><div class="eyebrow">FROM IMAGE TO INSIGHT</div>'
        '<h2>What happens when you classify?</h2><p>One example, three views of the same analysis.</p></div>'
        '<div class="landing-steps"><article class="preview-card"><div class="step-number">01 / FIND</div>'
        '<h3>Locate a region</h3>' + image("source.jpg", "Original blade inspection image before detection")
        + '<p>Upload a full image and detect candidate regions, or draw your own rectangle. '
        'You choose the area to inspect.</p></article>'
        '<article class="preview-card"><div class="step-number">02 / CLASSIFY</div>'
        '<h3>' + escape(HUMAN_LABELS[prediction["predicted_label"]]) + '</h3>'
        '<div class="crop-preview">' + image("crop.png", "Exact 224 by 224 RGB input for the example prediction")
        + f'<span>Selected region {escape(prediction["proposal_id"])}<br><strong>224 × 224 model input</strong></span></div>' + bars
        + '<p>The classifier scores six categories for the selected crop. Scores are model outputs, '
        'not calibrated probabilities.</p></article>'
        '<article class="preview-card"><div class="step-number">03 / EXPLORE</div>'
        '<h3>Inspect model attention</h3>' + image("attention.png", "Grad-CAM overlay for the predicted category")
        + '<div class="attention-key"><span>Lower activation</span><i></i><span>Higher activation</span></div>'
        '<p>Grad-CAM highlights areas associated with the predicted category. It is an explanation aid, '
        'not a defect outline.</p></article></div>'
        '<div class="landing-categories"><strong>Six categories, one workflow</strong><div>' + categories + '</div></div>'
        '<p class="example-note">Precomputed example from the project dataset using the same frozen detector '
        'and classifier as the app. Your image will produce its own results.</p></section>'
    )
    st.markdown("""<style>
    html, [data-testid="stAppViewContainer"] { scroll-behavior:smooth; }
    #classification-workspace { scroll-margin-top:5rem; }
    .landing { margin:1.5rem 0 3rem; }
    .case-counter { text-align:center; color:#53758f; padding:.65rem 0; font-size:.75rem; letter-spacing:.06em; }
    .landing-intro { padding-top:1rem; margin-bottom:.75rem; }
    .st-key-landing_preview { margin-top:1.5rem; gap:0; background:#fff; border:1px solid #c9deed; border-radius:18px; overflow:hidden; box-shadow:0 20px 45px #174b7418; }
    .st-key-landing_image { position:relative; gap:0; }
    .st-key-landing_image img { width:100%; height:320px; object-fit:contain; background:#e9eff3; display:block; }
    .st-key-landing_previous, .st-key-landing_next { position:absolute; top:50%; transform:translateY(-50%); width:auto !important; z-index:2; }
    .st-key-landing_previous { left:.75rem; }
    .st-key-landing_next { right:.75rem; }
    .st-key-landing_previous button, .st-key-landing_next button { background:#0b3158; color:white; border:1px solid #ffffff80; width:2.75rem; height:2.75rem; min-height:2.75rem; padding:0; border-radius:50%; box-shadow:0 3px 12px #0b315833; }
    .st-key-landing_previous button p, .st-key-landing_next button p { font-size:1.5rem; line-height:1; }
    .st-key-landing_previous button:hover, .st-key-landing_next button:hover { background:#0b5e99; color:white; }
    .preview-caption { padding:.85rem 1rem; color:#53758f; font-size:.8rem; }
    .landing h1 { color:#0b3158; font-size:clamp(2.4rem,4.5vw,4.4rem); line-height:1.05; letter-spacing:-.045em; margin:1rem 0; }
    .landing-intro p { color:#46677f; font-size:1.15rem; line-height:1.7; max-width:35rem; }
    a.classify-link { display:inline-flex; align-items:center; gap:2rem; background:#0b5e99; color:white; padding:.9rem 1.5rem; border-radius:10px; font-weight:750; text-decoration:none; margin:.8rem 0; }
    a.classify-link:hover { background:#084774; }
    a.classify-link:focus-visible { outline:3px solid #43a9df; outline-offset:4px; }
    .landing-meta { color:#53758f; font-size:.82rem; margin-top:.5rem; }
    .landing-figure { margin:0; background:#fff; border:1px solid #c9deed; border-radius:18px; overflow:hidden; box-shadow:0 20px 45px #174b7418; }
    .preview-toolbar { display:flex; align-items:center; gap:.5rem; padding:1rem; background:#0b3158; color:#fff; font-size:.65rem; letter-spacing:.06em; }
    .preview-toolbar > span:last-child { margin-left:auto; opacity:.7; }
    .preview-dot { width:.5rem; height:.5rem; background:#65dbbb; border-radius:50%; }
    .landing-figure img { width:100%; height:320px; object-fit:contain; background:#e9eff3; display:block; }
    .landing-figure figcaption { padding:.85rem 1rem; color:#53758f; font-size:.8rem; }
    .landing-section-label { margin:0 0 1.25rem; }
    .landing-section-label h2 { margin:.3rem 0; color:#0b3158; letter-spacing:-.025em; }
    .landing-section-label p { color:#53758f; margin:0; }
    .landing-steps { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1.2rem; }
    .preview-card { padding:1.3rem; border:1px solid #c9deed; background:#ffffffd9; border-radius:16px; }
    .step-number { color:#116eaf; font-size:.72rem; font-weight:800; letter-spacing:.08em; }
    .preview-card h3 { margin:.5rem 0 1rem; font-size:1.3rem; color:#102a43; }
    .preview-card > img { width:100%; height:230px; object-fit:contain; background:#edf3f7; border-radius:8px; }
    .preview-card p { color:#53758f; font-size:.88rem; line-height:1.6; margin:1rem 0 0; }
    .crop-preview { display:flex; align-items:center; gap:.8rem; margin-bottom:1rem; color:#53758f; font-size:.75rem; }
    .crop-preview img { width:64px; height:64px; border-radius:8px; }
    .preview-score { margin:.45rem 0; font-size:.75rem; }
    .preview-score > div:first-child { display:flex; justify-content:space-between; gap:1rem; }
    .preview-track { height:5px; background:#e1edf5; border-radius:9px; margin-top:.25rem; overflow:hidden; }
    .preview-track span { display:block; height:100%; background:#1987bf; border-radius:9px; }
    .attention-key { display:flex; align-items:center; gap:.5rem; font-size:.65rem; color:#53758f; margin:.6rem 0; }
    .attention-key i { flex:1; height:5px; background:linear-gradient(90deg,#2539bc,#38bcba,#ffdf50,#ed3434); border-radius:9px; }
    .landing-categories { padding:1.4rem 0 1rem; margin-top:1rem; color:#234e6d; }
    .landing-categories > div { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.7rem; }
    .landing-categories span { border:1px solid #c9deed; background:#fff; padding:.4rem .8rem; border-radius:999px; font-size:.8rem; }
    .example-note { color:#53758f; font-size:.75rem; }
    @media(max-width:900px) { .landing-intro { grid-template-columns:1fr; gap:1.5rem; } .landing-steps { grid-template-columns:1fr; } .preview-card > img { height:260px; } }
    @media(prefers-reduced-motion:reduce) { html, [data-testid="stAppViewContainer"] { scroll-behavior:auto; } }
    </style>""", unsafe_allow_html=True)
    copy, visual = st.columns([1.05, 1], gap="large")
    with copy:
        st.markdown(introduction, unsafe_allow_html=True)
    with visual:
        with st.container(key="landing_preview"):
            st.markdown(preview, unsafe_allow_html=True)
            with st.container(key="landing_image"):
                st.markdown(image("regions.jpg", "Real blade image with outlined detector proposals"), unsafe_allow_html=True)
                st.button("←", key="landing_previous", help="Previous example", on_click=change_case, args=(-1,))
                st.button("→", key="landing_next", help="Next example", on_click=change_case, args=(1,))
            st.markdown('<div class="preview-caption">Real detector output. Review the outlined regions before classification.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="case-counter">EXAMPLE {index + 1} OF {len(cases)} · BLADE IMAGE {cases[index]["id"]}</div>', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)
