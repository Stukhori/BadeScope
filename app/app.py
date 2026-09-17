"""Application v3: reviewed detector proposals plus the frozen classifier workflows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_cropper import st_cropper

from windblade_demo.constants import (
    APPLICATION_VERSION, CHECKPOINT_STATE_FINGERPRINT, CLASS_DESCRIPTIONS,
    CLASS_LABELS, HUMAN_LABELS, MODEL_DISPLAY_NAME, PREPROCESSING_CONTRACT,
)
from windblade_demo.crops import (
    SelectionValidationError, annotated_selection, contextual_crop, display_image,
    map_display_box, prepare_region,
)
from windblade_demo.detection_status import DetectorUnavailableError, load_detection_status
from windblade_demo.detector import (
    ZERO_PROPOSAL_MESSAGE, ProposalDetectorError, RegionProposal,
    load_proposal_detector, propose_regions, reviewed_proposals,
)
from windblade_demo.explain import generate_gradcam
from windblade_demo.exports import annotated_image_export, csv_export, json_export
from windblade_demo.inference import FrozenModelError, infer, load_frozen_model
from windblade_demo.inputs import UploadValidationError, decode_upload
from windblade_demo.research import FrozenResearchError, load_phase10
from windblade_demo.session import (
    RegionRecord, make_region_record, remove_region, replace_region, with_gradcam,
)
from windblade_demo.visualization import annotate_proposals, annotate_regions


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = (
    "Home", "Analyze Image", "Compare Regions", "Research Results",
    "Detection Readiness", "About",
)
ANALYSIS_MODES = (
    "Auto detection", "Prepared crop", "Manual single region", "Manual multi-region",
)

st.set_page_config(
    page_title="BladeScope | Wind Turbine Blade Defect Recognition", page_icon="🌬️", layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    .stApp {
        color: #102a43;
        background:
            radial-gradient(circle at 88% 4%, rgba(112, 190, 238, .20), transparent 24rem),
            linear-gradient(180deg, #edf6fd 0%, #f8fbfe 38%, #f2f7fb 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e5f1fb 0%, #f4f8fc 68%, #eaf3fb 100%);
        border-right: 1px solid #c8ddec;
    }
    .block-container {
        box-sizing: border-box;
        width: 100%;
        max-width: 1600px;
        padding: 1.2rem clamp(1rem, 2.25vw, 2.75rem) 3rem;
    }
    .hero { position: relative; overflow: hidden; padding: 1.65rem 1.8rem;
            border-radius: 22px; color: white;
            background: linear-gradient(125deg, #0b3158 0%, #0d5f9f 58%, #49a4dc 100%);
            box-shadow: 0 16px 38px rgba(18, 76, 122, .22); margin-bottom: 1rem; }
    .hero::before { content: ""; position: absolute; width: 19rem; height: 19rem;
                    right: -5rem; top: -11rem; border-radius: 50%;
                    border: 2px solid rgba(255,255,255,.20);
                    box-shadow: 0 0 0 2.7rem rgba(255,255,255,.055),
                                0 0 0 5.6rem rgba(255,255,255,.035); }
    .hero::after { content: ""; position: absolute; width: 22rem; height: 4rem;
                   right: 2rem; bottom: -2.1rem; border-radius: 50%;
                   border-top: 2px solid rgba(255,255,255,.24);
                   transform: rotate(-7deg); }
    .hero > * { position: relative; z-index: 1; }
    .hero h1 { margin: .2rem 0 0; font-size: clamp(1.85rem, 4vw, 2.6rem); line-height: 1.1; }
    .hero p { margin: .65rem 0 0; max-width: 840px; opacity: .94; font-size: 1.02rem; }
    .hero-copy { max-width: calc(100% - 12rem); }
    .turbine-scene { position: absolute; z-index: 0; right: 1.6rem; bottom: -.35rem;
                     width: 10.5rem; height: 10.5rem; opacity: .88; }
    .turbine-scene svg { width: 100%; height: 100%; overflow: visible; }
    .badge { display: inline-block; padding: .28rem .58rem; margin: 0 .32rem .3rem 0;
             border-radius: 999px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
             font-size: .72rem; font-weight: 750; letter-spacing: .055em; }
    .notice { border-left: 5px solid #2b8fd2; background: #eaf5fd; padding: .9rem 1rem;
              border-radius: 10px; margin: .75rem 0 1.2rem; color: #153d5c; }
    .card { height: 100%; box-sizing: border-box; background: rgba(255,255,255,.90);
            border: 1px solid #c9deed; border-radius: 15px;
            padding: 1rem 1.15rem; box-shadow: 0 6px 20px rgba(23,75,116,.09); }
    .workspace-status { border-top: 4px solid #2b8fd2; }
    .workspace-status h2 { margin: .35rem 0 .65rem; font-size: 1.35rem; }
    .status-row { display:flex; justify-content:space-between; gap:1rem; padding:.55rem 0;
                  border-top:1px solid #d9e8f3; color:#46677f; }
    .status-row strong { color:#102a43; text-align:right; }
    .eyebrow { color: #116eaf; font-size: .76rem; font-weight: 800; letter-spacing: .08em; }
    .status-ok { border-left: 4px solid #2589c9; background: #e8f4fc; padding: .8rem 1rem; border-radius: 10px; }
    .upload-ready {
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
        margin: .2rem 0 .85rem; padding: .72rem .9rem;
        border: 1px solid #b9dbea; border-radius: 11px;
        color: #234e6d; background: rgba(234,245,253,.88);
        animation: bladescope-fade .28s ease-out both;
    }
    .upload-ready-main { min-width: 0; display: flex; align-items: center; gap: .6rem; }
    .upload-ready-dot { width: .62rem; height: .62rem; flex: 0 0 auto; border-radius: 50%;
                        background: #1f9d79; box-shadow: 0 0 0 .25rem rgba(31,157,121,.12); }
    .upload-ready-name { overflow: hidden; color: #102a43; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
    .upload-ready-meta { flex: 0 0 auto; color: #53758f; font-size: .82rem; font-variant-numeric: tabular-nums; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.90); border:1px solid #c9deed; padding:.7rem; border-radius:12px; }
    div[data-testid="stFileUploader"] { background:rgba(255,255,255,.90); border:1px solid #c9deed; border-radius:14px; padding:.55rem .8rem; }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    div.stButton > button:hover:not(:disabled), div.stDownloadButton > button:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(18, 76, 122, .16);
    }
    div.stButton > button:active:not(:disabled), div.stDownloadButton > button:active:not(:disabled) {
        transform: translateY(0);
        box-shadow: 0 3px 8px rgba(18, 76, 122, .12);
    }
    div[data-testid="stFileUploader"] {
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }
    div[data-testid="stFileUploader"]:focus-within {
        border-color: #2b8fd2;
        box-shadow: 0 0 0 3px rgba(43,143,210,.16);
        transform: translateY(-1px);
    }
    [data-testid="stSpinner"] {
        position: relative; overflow: hidden; padding: .8rem 1rem;
        border: 1px solid #c9deed; border-radius: 12px;
        background: rgba(255,255,255,.88);
        box-shadow: 0 8px 22px rgba(23,75,116,.09);
    }
    [data-testid="stSpinner"]::after {
        content: ""; position: absolute; left: 0; bottom: 0; height: 3px; width: 42%;
        border-radius: 999px; background: linear-gradient(90deg, #2b8fd2, #62c3ef);
        animation: bladescope-loading 1.25s ease-in-out infinite alternate;
    }
    .hero { animation: bladescope-rise .45s ease-out both; }
    .card { animation: bladescope-rise .42s ease-out both; }
    div[data-testid="stMetric"], div[data-testid="stDataFrame"], div[data-testid="stImage"] {
        animation: bladescope-fade .32s ease-out both;
    }
    @keyframes bladescope-rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes bladescope-fade {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes bladescope-loading {
        from { transform: translateX(-12%); }
        to { transform: translateX(150%); }
    }
    .brand-lockup { display:flex; align-items:center; gap:.75rem; padding:.35rem .2rem .8rem; }
    .brand-mark { display:grid; place-items:center; width:2.8rem; height:2.8rem; border-radius:14px;
                  color:white; background:linear-gradient(145deg,#0b4f83,#2b98d4);
                  box-shadow:0 7px 18px rgba(19,91,139,.22); }
    .brand-mark svg { width:2rem; height:2rem; }
    .brand-name { color:#0b3158; font-size:1.4rem; font-weight:800; line-height:1; letter-spacing:-.02em; }
    .brand-subtitle { color:#53758f; font-size:.72rem; margin-top:.25rem; letter-spacing:.08em; text-transform:uppercase; }
    @media (max-width: 700px) {
        .block-container { padding: .8rem .8rem 2.25rem; }
        .hero { padding:1.15rem; border-radius:14px; }
        .hero-copy { max-width:100%; }
        .turbine-scene { display:none; }
        .card { padding:.9rem 1rem; }
        .status-row { align-items:flex-start; }
        .upload-ready { align-items:flex-start; flex-direction:column; gap:.25rem; }
    }
    @media (max-width: 480px) {
        .block-container { padding-top: 4rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        .hero, .card, div[data-testid="stMetric"], div[data-testid="stDataFrame"],
        div[data-testid="stImage"], .upload-ready, [data-testid="stSpinner"]::after {
            animation: none !important;
        }
        div.stButton > button, div.stDownloadButton > button,
        div[data-testid="stFileUploader"] { transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Verifying and loading the frozen checkpoint…")
def cached_model():
    return load_frozen_model(ROOT)


@st.cache_data(show_spinner=False)
def cached_research():
    return load_phase10(ROOT)


@st.cache_data(show_spinner=False)
def cached_detection_status():
    return load_detection_status(ROOT)


@st.cache_resource(show_spinner="Verifying and loading the frozen proposal detector…")
def cached_proposal_detector():
    return load_proposal_detector(ROOT)


@contextmanager
def analysis_status(label: str, completed_label: str) -> Iterator[Any]:
    """Show consistent feedback around a user-triggered analysis."""
    status = st.status(label, expanded=True)
    status.write("Preparing the image and checking the required model assets.")
    try:
        yield status
    except Exception:
        status.update(label="Analysis stopped", state="error", expanded=True)
        raise
    else:
        status.update(label=completed_label, state="complete", expanded=False)


def initialize_session() -> None:
    st.session_state.setdefault("analysis_records", [])
    st.session_state.setdefault("source_images", {})
    st.session_state.setdefault("multi_uploader_nonce", 0)
    st.session_state.setdefault("next_region_number", 1)
    st.session_state.setdefault("proposal_results", {})


def records() -> list[RegionRecord]:
    return st.session_state["analysis_records"]


def save_source(decoded: Any) -> None:
    st.session_state["source_images"][decoded.byte_sha256] = decoded.image.copy()


def render_upload_summary(decoded: Any) -> None:
    """Confirm the validated upload without exposing unsafe filename markup."""
    width, height = decoded.image.size
    size_kib = decoded.byte_count / 1024
    st.markdown(
        '<div class="upload-ready" role="status">'
        '<div class="upload-ready-main"><span class="upload-ready-dot" aria-hidden="true"></span>'
        f'<span class="upload-ready-name">{escape(decoded.filename)}</span></div>'
        f'<span class="upload-ready-meta">{width} × {height} px · {size_kib:,.1f} KiB</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def add_record(record: RegionRecord) -> None:
    st.session_state["analysis_records"] = [*records(), record]


def score_rows(record: RegionRecord) -> list[dict[str, Any]]:
    return [
        {"Category": HUMAN_LABELS[label], "Model score": record.scores[index]}
        for index, label in enumerate(CLASS_LABELS)
    ]


def render_scores(record: RegionRecord, *, key: str) -> None:
    rows = score_rows(record)
    st.subheader(HUMAN_LABELS[record.predicted_label])
    st.caption(f"{record.region_id} · classifier scores")
    if record.detector_confidence is not None:
        st.metric("Detector confidence", f"{record.detector_confidence:.6f}")
        st.caption("Detector confidence is shown separately from the classifier scores below.")
    st.vega_lite_chart(
        data=rows,
        spec={"mark": {"type": "bar", "cornerRadiusEnd": 4, "color": "#1677c8"},
         "encoding": {
             "x": {"field": "Model score", "type": "quantitative", "scale": {"domain": [0, 1]}},
             "y": {"field": "Category", "type": "nominal", "sort": "-x"},
             "tooltip": ["Category", {"field": "Model score", "format": ".6f"}]},
         "height": 210},
        width="stretch", key=f"scores_{key}",
    )


def classify_record(
    *, mode: str, decoded: Any, model_input,
    selected_box: tuple[int, int, int, int] | None = None,
    contextual_box: tuple[int, int, int, int] | None = None,
    replacement_id: str | None = None,
    detector_proposal_id: str | None = None,
    detector_confidence: float | None = None,
) -> RegionRecord:
    result = infer(cached_model(), model_input)
    chosen_id = replacement_id
    if chosen_id is None:
        chosen_id = f"R{st.session_state['next_region_number']}"
    record = make_region_record(
        records=records(), mode=mode, source_name=decoded.filename,
        source_sha256=decoded.byte_sha256, source_size=decoded.image.size,
        model_input=model_input, result=result, selected_box=selected_box,
        contextual_box=contextual_box, region_id=chosen_id,
        detector_proposal_id=detector_proposal_id,
        detector_confidence=detector_confidence,
    )
    if replacement_id is None:
        st.session_state["next_region_number"] += 1
    return record


def render_hero(title: str, description: str) -> None:
    st.markdown(
        '<div class="hero"><div class="hero-copy"><div><span class="badge">BLADESCOPE</span>'
        '<span class="badge">WIND TURBINE VISION</span>'
        '<span class="badge">LOCAL PROCESSING</span><span class="badge">VERIFIED CLASSIFIER</span></div>'
        f'<h1>{title}</h1><p>{description}</p></div>'
        '<div class="turbine-scene" aria-hidden="true"><svg viewBox="0 0 180 180" fill="none">'
        '<path d="M91 72L83 180H99L93 72" fill="rgba(255,255,255,.82)"/>'
        '<circle cx="92" cy="66" r="7" fill="white"/>'
        '<path d="M89 61C75 43 67 25 70 10C84 25 91 42 94 59Z" fill="rgba(255,255,255,.88)"/>'
        '<path d="M99 66C120 62 139 64 151 74C132 79 113 76 98 71Z" fill="rgba(255,255,255,.76)"/>'
        '<path d="M88 72C81 92 70 108 56 115C58 96 68 80 84 68Z" fill="rgba(255,255,255,.68)"/>'
        '<path d="M20 161C53 151 124 151 164 163" stroke="rgba(255,255,255,.35)" stroke-width="2"/>'
        '</svg></div></div>', unsafe_allow_html=True,
    )


def render_scope_notice() -> None:
    st.markdown(
        '<div class="notice"><strong>Region-based analysis.</strong> '
        'Use auto detection to find regions, or choose a prepared crop or manual rectangle. '
        'Selected regions are then classified into the six blade-defect categories.</div>',
        unsafe_allow_html=True,
    )


def go_to(page: str) -> None:
    st.session_state["navigation"] = page


def go_to_analysis(mode: str) -> None:
    st.session_state["analysis_mode"] = mode
    st.session_state["navigation"] = "Analyze Image"


def render_home() -> None:
    render_hero(
        "Detect and classify blade defects",
        "Upload a blade image, detect likely defect regions automatically, review the results, and classify each selected region.",
    )
    render_scope_notice()
    primary_workflow, workspace_status = st.columns([1.7, 1], gap="large")
    with primary_workflow:
        st.markdown(
            '<div class="card"><div class="eyebrow">PRIMARY WORKFLOW</div>'
            '<h2>Auto detection</h2><p>Find likely defect regions, review the numbered boxes, '
            'and classify the regions you select.</p></div>',
            unsafe_allow_html=True,
        )
        st.button(
            "Detect and classify defects", type="primary", width="stretch",
            on_click=go_to_analysis, args=("Auto detection",),
        )
    with workspace_status:
        st.markdown(
            '<div class="card workspace-status"><div class="eyebrow">WORKSPACE STATUS</div>'
            '<h2>Ready for local analysis</h2>'
            '<div class="status-row"><span>Auto detection</span><strong>Ready</strong></div>'
            '<div class="status-row"><span>Analysis modes</span><strong>4</strong></div>'
            f'<div class="status-row"><span>Saved regions</span><strong>{len(records())}</strong></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Additional analysis features")
    first, second, third = st.columns(3)
    with first:
        st.markdown('<div class="card"><div class="eyebrow">PREPARED</div><h3>Classify a crop</h3><p>Use an image already centered on one visible region.</p></div>', unsafe_allow_html=True)
        st.button("Analyze prepared crop", width="stretch", on_click=go_to_analysis, args=("Prepared crop",))
    with second:
        st.markdown('<div class="card"><div class="eyebrow">SINGLE</div><h3>Draw one region</h3><p>Select and classify one rectangle on a larger image.</p></div>', unsafe_allow_html=True)
        st.button("Analyze one manual region", width="stretch", on_click=go_to_analysis, args=("Manual single region",))
    with third:
        st.markdown('<div class="card"><div class="eyebrow">MULTI</div><h3>Build a region set</h3><p>Add, replace, compare, and export multiple manual regions.</p></div>', unsafe_allow_html=True)
        st.button("Analyze multiple regions", width="stretch", on_click=go_to_analysis, args=("Manual multi-region",))
    action_a, action_b, action_c = st.columns(3)
    action_a.button("Compare saved regions", width="stretch", on_click=go_to, args=("Compare Regions",))
    action_b.button("Open research results", width="stretch", on_click=go_to, args=("Research Results",))
    action_c.button("Check detection readiness", width="stretch", on_click=go_to, args=("Detection Readiness",))


def render_prepared() -> None:
    st.subheader("Prepared crop classification")
    st.caption("Upload a crop already centered on one visible region for classification.")
    uploaded = st.file_uploader("Choose a prepared PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="prepared_v2")
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    render_upload_summary(decoded)
    model_input = prepare_region(decoded.image)
    left, right = st.columns(2)
    left.image(decoded.image, caption="Uploaded crop", width="stretch")
    right.image(model_input, caption="Exact RGB 224×224 model input", width="stretch")
    if st.button("Classify and add prepared crop", type="primary", key="prepared_classify"):
        with analysis_status("Classifying the prepared crop…", "Prepared crop classified"):
            record = classify_record(mode="prepared_crop", decoded=decoded, model_input=model_input)
            add_record(record)
            st.session_state["latest_region_id"] = record.region_id
    latest = next((item for item in reversed(records()) if item.region_id == st.session_state.get("latest_region_id")), None)
    if latest and latest.source_sha256 == decoded.byte_sha256:
        render_scores(latest, key=f"prepared_{latest.region_id}")


def manual_selection(decoded: Any, *, key: str):
    displayed = display_image(decoded.image)
    rectangle = st_cropper(
        displayed, realtime_update=True, box_color="#1D4ED8", aspect_ratio=None,
        return_type="box", should_resize_image=False, stroke_width=3, key=key,
    )
    selected = map_display_box(rectangle, display_size=displayed.size, original_size=decoded.image.size)
    crop = contextual_crop(decoded.image, selected)
    st.caption(
        f"Display {displayed.width}×{displayed.height} → original {decoded.image.width}×{decoded.image.height}; "
        f"selected original box {selected.as_tuple()}."
    )
    return selected, crop


def render_manual_single() -> None:
    st.subheader("Manual single-region classification")
    st.caption("Upload a larger image and draw one rectangle around a visible region.")
    uploaded = st.file_uploader("Choose a larger PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="single_v2")
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    render_upload_summary(decoded)
    selected, crop = manual_selection(decoded, key=f"single_cropper_{decoded.byte_sha256[:12]}")
    left, right = st.columns([1.35, 1])
    left.image(annotated_selection(decoded.image, crop), caption="Cobalt: your rectangle · Sky blue: contextual crop", width="stretch")
    right.image(crop.model_input, caption="Exact contextual RGB 224×224 model input", width="stretch")
    if st.button("Classify and add selected region", type="primary", key="single_classify"):
        with analysis_status("Classifying the selected region…", "Selected region classified"):
            geometry = crop.geometry
            record = classify_record(
                mode="manual_single_region", decoded=decoded, model_input=crop.model_input,
                selected_box=selected.as_tuple(),
                contextual_box=(geometry.crop_xmin, geometry.crop_ymin, geometry.crop_xmax, geometry.crop_ymax),
            )
            add_record(record)
            st.session_state["latest_region_id"] = record.region_id
    latest = next((item for item in reversed(records()) if item.region_id == st.session_state.get("latest_region_id")), None)
    if latest and latest.source_sha256 == decoded.byte_sha256 and latest.selected_box == selected.as_tuple():
        render_scores(latest, key=f"single_{latest.region_id}")


def render_manual_multi() -> None:
    st.subheader("Manual multi-region analysis")
    st.caption("Draw one rectangle at a time. Saved regions receive stable IDs (R1, R2, …), may overlap, and are classified independently.")
    top_left, top_right = st.columns([3, 1])
    with top_right:
        if st.button("New image", width="stretch", key="multi_new_image"):
            st.session_state["multi_uploader_nonce"] += 1
            st.session_state.pop("multi_active_hash", None)
            st.rerun()
    with top_left:
        uploaded = st.file_uploader(
            "Choose one source image", type=["png", "jpg", "jpeg"],
            key=f"multi_v2_{st.session_state['multi_uploader_nonce']}",
        )
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    render_upload_summary(decoded)
    st.session_state["multi_active_hash"] = decoded.byte_sha256
    source_records = [item for item in records() if item.source_sha256 == decoded.byte_sha256 and item.mode == "manual_multi_region"]
    if source_records:
        st.image(annotate_regions(decoded.image, source_records), caption=f"Saved manual regions: {len(source_records)}", width="stretch")
    selected, crop = manual_selection(decoded, key=f"multi_cropper_{decoded.byte_sha256[:12]}")
    preview_left, preview_right = st.columns([1.4, 1])
    preview_left.image(annotated_selection(decoded.image, crop), caption="Current unsaved rectangle and contextual crop", width="stretch")
    preview_right.image(crop.model_input, caption="Current exact model input", width="stretch")
    geometry = crop.geometry
    contextual_box = (geometry.crop_xmin, geometry.crop_ymin, geometry.crop_xmax, geometry.crop_ymax)
    action_left, action_mid, action_right = st.columns(3)
    if action_left.button("Add and classify region", type="primary", width="stretch", key="multi_add"):
        with analysis_status("Classifying and saving this region…", "Region classified and saved"):
            record = classify_record(
                mode="manual_multi_region", decoded=decoded, model_input=crop.model_input,
                selected_box=selected.as_tuple(), contextual_box=contextual_box,
            )
            add_record(record)
            st.session_state["latest_region_id"] = record.region_id
            st.rerun()
    selected_id = action_mid.selectbox(
        "Saved region", [item.region_id for item in source_records], disabled=not source_records,
        key=f"multi_selected_{decoded.byte_sha256[:8]}",
    ) if source_records else None
    if action_right.button("Replace with current rectangle", width="stretch", disabled=not selected_id, key="multi_replace"):
        with analysis_status("Reclassifying the replacement region…", "Saved region replaced"):
            replacement = classify_record(
                mode="manual_multi_region", decoded=decoded, model_input=crop.model_input,
                selected_box=selected.as_tuple(), contextual_box=contextual_box, replacement_id=selected_id,
            )
            st.session_state["analysis_records"] = replace_region(records(), replacement)
            st.session_state["latest_region_id"] = replacement.region_id
            st.rerun()
    remove_left, clear_right = st.columns(2)
    if remove_left.button("Remove selected region", disabled=not selected_id, width="stretch", key="multi_remove"):
        st.session_state["analysis_records"] = remove_region(records(), selected_id)
        st.rerun()
    if clear_right.button("Clear regions for this image", disabled=not source_records, width="stretch", key="multi_clear"):
        st.session_state["analysis_records"] = [
            item for item in records()
            if not (item.source_sha256 == decoded.byte_sha256 and item.mode == "manual_multi_region")
        ]
        st.rerun()


def proposal_table(proposals: tuple[RegionProposal, ...]) -> list[dict[str, Any]]:
    return [
        {
            "Region": proposal.proposal_id,
            "Detected result": "Defect region",
            "Detector confidence": proposal.detector_confidence,
            "Box": proposal.box.as_tuple(),
        }
        for proposal in proposals
    ]


def render_automatic_proposals() -> None:
    st.subheader("Auto detection")
    st.caption("Upload a blade image to detect regions, then choose which ones to classify.")
    uploaded = st.file_uploader(
        "Choose a full PNG, JPG, or JPEG image", type=["png", "jpg", "jpeg"], key="automatic_v3"
    )
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    render_upload_summary(decoded)
    proposal_store = st.session_state["proposal_results"]
    if st.button("Detect defect regions", type="primary", key="automatic_generate"):
        with analysis_status("Scanning the blade image for defect regions…", "Region detection complete"):
            proposal_store[decoded.byte_sha256] = propose_regions(
                cached_proposal_detector(), decoded.image
            )
    if decoded.byte_sha256 not in proposal_store:
        st.image(decoded.image, caption="Uploaded image", width="stretch")
        return

    proposals = tuple(proposal_store[decoded.byte_sha256])
    if not proposals:
        st.info(ZERO_PROPOSAL_MESSAGE)
        st.caption("You can also select a region manually.")
        left, right = st.columns(2)
        left.button(
            "Use manual single-region workflow", width="stretch", on_click=go_to_analysis,
            args=("Manual single region",), key="zero_manual_single",
        )
        right.button(
            "Use manual multi-region workflow", width="stretch", on_click=go_to_analysis,
            args=("Manual multi-region",), key="zero_manual_multi",
        )
        return

    st.image(
        annotate_proposals(decoded.image, proposals),
        caption=f"{len(proposals)} detected region(s)",
        width="stretch",
    )
    st.dataframe(proposal_table(proposals), hide_index=True, width="stretch")
    proposal_ids = [proposal.proposal_id for proposal in proposals]
    selected_ids = st.multiselect(
        "Select detected regions to classify", proposal_ids, key=f"proposal_select_{decoded.byte_sha256[:12]}"
    )
    confirmed = st.checkbox(
        "I reviewed the selected regions", key=f"proposal_review_{decoded.byte_sha256[:12]}"
    )
    if st.button(
        "Classify selected regions", type="primary", key="automatic_classify",
        disabled=not selected_ids or not confirmed,
    ):
        accepted = reviewed_proposals(proposals, selected_ids, reviewed=confirmed)
        with analysis_status("Classifying the reviewed regions…", "Reviewed regions classified"):
            progress = st.progress(0.0, text=f"Classifying 0 of {len(accepted)} regions")
            for index, proposal in enumerate(accepted, start=1):
                crop = contextual_crop(decoded.image, proposal.box)
                geometry = crop.geometry
                record = classify_record(
                    mode="auto_detection",
                    decoded=decoded,
                    model_input=crop.model_input,
                    selected_box=proposal.box.as_tuple(),
                    contextual_box=(
                        geometry.crop_xmin, geometry.crop_ymin,
                        geometry.crop_xmax, geometry.crop_ymax,
                    ),
                    detector_proposal_id=proposal.proposal_id,
                    detector_confidence=proposal.detector_confidence,
                )
                add_record(record)
                st.session_state["latest_region_id"] = record.region_id
                progress.progress(
                    index / len(accepted), text=f"Classifying {index} of {len(accepted)} regions"
                )
        st.rerun()

    accepted_records = [
        item for item in records()
        if item.source_sha256 == decoded.byte_sha256
        and item.mode == "auto_detection"
    ]
    if accepted_records:
        st.markdown("### Classified regions")
        for record in accepted_records:
            render_scores(record, key=f"automatic_{record.region_id}")


def render_analyze() -> None:
    render_hero("Detect and classify", "Automatically detect defect regions or choose one of the additional classification workflows.")
    render_scope_notice()
    mode = st.radio("Analysis mode", ANALYSIS_MODES, horizontal=True, key="analysis_mode")
    try:
        if mode == "Auto detection":
            render_automatic_proposals()
        elif mode == "Prepared crop":
            render_prepared()
        elif mode == "Manual single region":
            render_manual_single()
        else:
            render_manual_multi()
    except (
        UploadValidationError, SelectionValidationError, FrozenModelError,
        ProposalDetectorError, RuntimeError,
    ) as exc:
        st.error(str(exc))


def render_compare() -> None:
    render_hero("Compare regions", "Inspect every region saved in this browser session and export reproducible metadata without server-side persistence.")
    items = records()
    if not items:
        st.info("No regions are saved yet. Open Analyze Image and classify a user-supplied region first.")
        st.button("Go to Analyze Image", type="primary", on_click=go_to, args=("Analyze Image",))
        return
    sort_by = st.selectbox("Sort comparison", ("Region ID", "Top score", "Prediction"), key="compare_sort")
    if sort_by == "Top score":
        items = sorted(items, key=lambda item: max(item.scores), reverse=True)
    elif sort_by == "Prediction":
        items = sorted(items, key=lambda item: (HUMAN_LABELS[item.predicted_label], int(item.region_id[1:])))
    else:
        items = sorted(items, key=lambda item: int(item.region_id[1:]))
    rows = [
        {"Region": item.region_id, "Mode": item.mode.replace("_", " "), "Source": item.source_name,
         "Prediction": HUMAN_LABELS[item.predicted_label], "Top score": max(item.scores),
         "Detector confidence": item.detector_confidence,
         "Grad-CAM": item.gradcam_status}
        for item in items
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    chart_rows = [
        {"Region": item.region_id, "Category": HUMAN_LABELS[label], "Model score": item.scores[index]}
        for item in items for index, label in enumerate(CLASS_LABELS)
    ]
    st.vega_lite_chart(
        data=chart_rows,
        spec={"mark": "bar", "encoding": {
            "x": {"field": "Region", "type": "nominal"},
            "y": {"field": "Model score", "type": "quantitative"},
            "color": {"field": "Category", "type": "nominal"},
            "xOffset": {"field": "Category"},
            "tooltip": ["Region", "Category", {"field": "Model score", "format": ".6f"}]},
         "height": 300}, width="stretch",
    )
    selected_id = st.selectbox("Inspect region", [item.region_id for item in items], key="compare_selected")
    selected = next(item for item in items if item.region_id == selected_id)
    image_col, scores_col = st.columns([1, 1.7])
    image_col.image(selected.thumbnail, caption=f"{selected.region_id} exact model input", width="stretch")
    with scores_col:
        render_scores(selected, key=f"compare_{selected.region_id}")
    st.caption(f"Selected box: {selected.selected_box or 'prepared crop'} · contextual box: {selected.contextual_box or 'not applicable'}")
    if selected.detector_confidence is not None:
        st.caption(
            f"Detected region {selected.detector_proposal_id} confidence: "
            f"{selected.detector_confidence:.6f}; this is separate from classifier scores."
        )
    if st.button("Generate Grad-CAM for selected region", key="compare_gradcam"):
        with analysis_status("Generating the activation visualization…", "Activation visualization ready"):
            visual = generate_gradcam(cached_model(), selected.model_input, selected.predicted_class_id)
            st.session_state["analysis_records"] = replace_region(items, with_gradcam(selected, visual.overlay))
            st.rerun()
    if selected.gradcam_overlay is not None:
        st.info("Grad-CAM highlights the image areas that influenced the crop classification.")
        st.image(selected.gradcam_overlay, caption=f"{selected.region_id} Grad-CAM overlay", width=420)
    st.markdown("### Session exports")
    st.caption("Exports are generated in memory when requested. Uploaded images and analysis records are not written to the server.")
    export_a, export_b, export_c = st.columns(3)
    export_a.download_button("Download JSON", json_export(items), "blade-session.json", "application/json", width="stretch")
    export_b.download_button("Download CSV", csv_export(items), "blade-session.csv", "text/csv", width="stretch")
    manual_sources = sorted({item.source_sha256 for item in items if item.selected_box is not None})
    if manual_sources:
        source_hash = export_c.selectbox(
            "Annotated source", manual_sources,
            format_func=lambda value: next(item.source_name for item in items if item.source_sha256 == value),
            key="export_source",
        )
        matching = [item for item in items if item.source_sha256 == source_hash and item.selected_box is not None]
        image = st.session_state["source_images"].get(source_hash)
        if image is not None:
            export_c.download_button(
                "Download annotated PNG", annotated_image_export(image, matching),
                "blade-manual-regions.png", "image/png", width="stretch",
            )
    controls_a, controls_b = st.columns(2)
    if controls_a.button("Remove inspected region", width="stretch"):
        st.session_state["analysis_records"] = remove_region(items, selected_id)
        st.rerun()
    if controls_b.button("Clear session", width="stretch"):
        st.session_state["analysis_records"] = []
        st.session_state["source_images"] = {}
        st.session_state["next_region_number"] = 1
        st.rerun()


def render_research() -> None:
    render_hero("Research results", "A read-only dashboard over verified benchmark tables. Values are loaded from checked artifacts and are never recomputed in the app.")
    try:
        research = cached_research()
    except FrozenResearchError as exc:
        st.error(str(exc))
        return
    st.markdown('<div class="status-ok"><strong>Research analysis complete and locked.</strong> Canonical source fingerprint verified.</div>', unsafe_allow_html=True)
    clean = research["tables"]["clean_method_comparison"]
    summary = research["summary"]
    a, b, c, d = st.columns(4)
    a.metric("Test instances", clean[0]["test_instances"])
    b.metric("Test source images", clean[0]["test_sources"])
    c.metric("Methods compared", str(len(clean)))
    d.metric("Bootstrap resamples", str(summary["bootstrap_resamples"]))
    st.markdown("### Clean held-out performance")
    clean_view = [
        {"Method": row["method_name"], "Macro F1": float(row["macro_f1"]),
         "95% bootstrap CI low": float(row["macro_f1_bootstrap_ci_low"]),
         "95% bootstrap CI high": float(row["macro_f1_bootstrap_ci_high"]),
         "Accuracy": float(row["accuracy"]), "Balanced accuracy": float(row["balanced_accuracy"])}
        for row in clean
    ]
    st.dataframe(clean_view, hide_index=True, width="stretch")
    st.bar_chart(clean_view, x="Method", y="Macro F1", horizontal=True)
    st.markdown("### Data efficiency")
    efficiency_view = [
        {"Method": row["method_name"], "Training fraction": float(row["training_fraction"]),
         "Macro F1": float(row["macro_f1_mean"])}
        for row in research["tables"]["data_efficiency_summary"]
    ]
    st.line_chart(efficiency_view, x="Training fraction", y="Macro F1", color="Method")
    st.markdown("### Robustness retention")
    robust_view = [
        {"Method": row["method_name"], "Condition": row["condition_id"],
         "Retention (%)": float(row["retention_percent"])}
        for row in research["tables"]["robustness_retention_summary"] if row["condition_id"] != "clean"
    ]
    st.dataframe(robust_view, hide_index=True, width="stretch")
    st.markdown("### Error and human-review summary")
    st.dataframe(research["tables"]["error_human_review_summary"], hide_index=True, width="stretch")
    st.info("These are descriptive research summaries, not live estimates for the uploaded image. Intervals and seed variability retain the documented evaluation definitions.")
    st.caption(f"Verified research-source fingerprint: {research['scientific_output_fingerprint']}")


def render_detection() -> None:
    render_hero("Auto detection", "Explore the image annotations and class coverage behind automatic defect-region detection.")
    try:
        status = cached_detection_status()
    except DetectorUnavailableError as exc:
        st.error(str(exc))
        return
    render_scope_notice()
    st.markdown("### Frozen dataset audit")
    audit = status.audit
    first, second, third, fourth = st.columns(4)
    first.metric("Curated images", audit["curated_image_count"])
    second.metric("Curated boxes", audit["curated_box_count"])
    third.metric("Multi-box images", audit["images_with_multiple_boxes"])
    fourth.metric("Defect categories", len(audit["classes"]))
    split_rows = [
        {"Split": split.title(), "Images": audit["split_image_counts"][split], "Boxes": audit["split_box_counts"][split]}
        for split in ("train", "validation", "test")
    ]
    st.dataframe(split_rows, hide_index=True, width="stretch")
    class_rows = [{"Class": HUMAN_LABELS.get(label, label), "Boxes": count} for label, count in audit["classes"].items()]
    st.bar_chart(class_rows, x="Class", y="Boxes", horizontal=True)
    duplicate_a, duplicate_b, duplicate_c = st.columns(3)
    duplicate_a.metric("Cross-split duplicate/related pairs", audit["cross_split_duplicate_or_related_pair_count"])
    duplicate_b.metric("Retained exact-duplicate groups", audit["retained_exact_duplicate_groups"])
    duplicate_c.metric("Invalid boxes", audit["suspicious_geometry"]["invalid_boxes"])
    provenance = audit["annotation_provenance"]
    st.caption(
        f"Annotation format: {audit['annotation_format']} · source: {provenance['dataset_name']} v{provenance['dataset_version']} · "
        f"license {provenance['license']} · dataset DOI {provenance['versioned_dataset_doi']}"
    )
    st.markdown("### Auto-detection workflow")
    st.markdown(
        "- The detector finds likely defect regions in a full blade image.\n"
        "- Numbered regions can be reviewed and selected for classification.\n"
        "- The classifier assigns one of six blade-defect categories to each selected region."
    )
    st.caption(f"Verified annotation-audit fingerprint: {status.scientific_output_fingerprint}")


def render_about() -> None:
    render_hero("About BladeScope", "Automatic defect-region detection and classification for wind-turbine blade images.")
    st.markdown("### Model information")
    st.write(MODEL_DISPLAY_NAME)
    st.code(f"Checkpoint state fingerprint: {CHECKPOINT_STATE_FINGERPRINT}\nPreprocessing: {PREPROCESSING_CONTRACT}")
    st.markdown("### Six output categories")
    st.dataframe(
        [{"Category": HUMAN_LABELS[label], "Brief dataset-label guide": CLASS_DESCRIPTIONS[label]} for label in CLASS_LABELS],
        hide_index=True, width="stretch",
    )
    st.caption("These descriptions are plain-language guides to the six blade-defect categories.")
    st.markdown("### How analysis works")
    st.markdown(
        "- Auto detection finds and numbers likely defect regions in a full image.\n"
        "- Select the detected regions you want to classify.\n"
        "- Prepared-crop and manual-region tools provide additional ways to classify an area.\n"
        "- Compare saved regions, inspect score charts, generate Grad-CAM views, and export session results."
    )
    st.markdown("### Privacy and persistence")
    st.info("Uploads, crops, session history, visualizations, and exports remain in process memory for the active session. The app makes no external API calls and does not persist uploads or analysis history.")
    st.markdown("### Research foundation")
    st.write("BladeScope combines a verified defect-region detector with a six-category crop classifier and read-only research results.")


initialize_session()
with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark" aria-hidden="true">'
        '<svg viewBox="0 0 40 40" fill="none"><path d="M20 18L17 38H23L21 18" fill="currentColor"/>'
        '<circle cx="20" cy="15" r="2.7" fill="currentColor"/>'
        '<path d="M19 13C14 8 12 4 13 1C18 5 20 9 21 13Z" fill="currentColor"/>'
        '<path d="M23 15C29 13 34 14 37 17C31 19 26 18 22 17Z" fill="currentColor"/>'
        '<path d="M18 17C16 23 13 27 9 29C9 23 12 19 17 15Z" fill="currentColor"/>'
        '</svg></div><div><div class="brand-name">BladeScope</div>'
        '<div class="brand-subtitle">Wind turbine vision</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", NAVIGATION, key="navigation")
    st.caption(f"Application v{APPLICATION_VERSION}")
    st.divider()
    st.metric("Session regions", len(records()))
    st.success("Auto detection and classification · CPU · local processing")
    if st.button("Clear all session data", width="stretch", disabled=not records()):
        st.session_state["analysis_records"] = []
        st.session_state["source_images"] = {}
        st.session_state["next_region_number"] = 1
        st.session_state["proposal_results"] = {}
        st.rerun()
    st.caption("PNG/JPG/JPEG · max 15 MB · no upload persistence or telemetry")

if page == "Home":
    render_home()
elif page == "Analyze Image":
    render_analyze()
elif page == "Compare Regions":
    render_compare()
elif page == "Research Results":
    render_research()
elif page == "Detection Readiness":
    render_detection()
else:
    render_about()
