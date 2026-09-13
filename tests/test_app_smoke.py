from __future__ import annotations

from pathlib import Path

import pytest

from windblade_demo.constants import HUMAN_LABELS


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app/app.py"
NAVIGATION = [
    "Home", "Analyze Image", "Compare Regions", "Research Results",
    "Detection Readiness", "About",
]


def button(app, label):
    return next(item for item in app.button if item.label == label)


def open_analysis(app):
    next(item for item in app.radio if item.label == "Navigation").set_value("Analyze Image").run(timeout=30)
    mode = next(item for item in app.radio if item.label == "Analysis mode")
    assert mode.options == [
        "Auto detection", "Prepared crop", "Manual single region", "Manual multi-region",
    ]
    mode.set_value("Prepared crop").run(timeout=30)
    return app


def chart_table(chart):
    pyarrow = pytest.importorskip("pyarrow")
    return pyarrow.ipc.open_stream(chart.proto.data.data).read_all()


def test_application_v2_starts_on_home_without_an_upload():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert app.radio[0].label == "Navigation"
    assert app.radio[0].options == NAVIGATION
    assert len(app.file_uploader) == 0


def test_application_v3_leads_with_auto_detection():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    rendered = "\n".join(element.value for element in app.markdown)
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Region-based analysis" in rendered
    assert "BladeScope" in rendered
    assert "Detect and classify blade defects" in rendered
    assert "Auto detection" in rendered
    assert 'class="turbine-scene"' in source
    assert 'class="brand-lockup"' in source
    assert "unavailable" not in rendered.lower()
    assert "experimental" not in source.lower()
    assert "CUDA" not in rendered
    assert "ultralytics" not in source.lower()


def test_application_v3_uses_the_available_workspace_width():
    source = APP_PATH.read_text(encoding="utf-8")
    container_css = source[source.index(".block-container {"):source.index(".hero {")]
    assert "box-sizing: border-box" in container_css
    assert "width: 100%" in container_css
    assert "max-width: 1600px" in container_css
    assert "clamp(1rem, 2.25vw, 2.75rem)" in container_css
    assert 'initial_sidebar_state="auto"' in source
    assert "@media (max-width: 700px)" in source
    assert ".block-container { padding: .8rem .8rem 2.25rem; }" in source
    assert "@media (max-width: 480px)" in source
    assert ".block-container { padding-top: 4rem; }" in source


def test_home_balances_primary_workflow_with_workspace_status():
    source = APP_PATH.read_text(encoding="utf-8")
    home = source[source.index("def render_home"):source.index("def render_prepared")]
    assert 'st.columns([1.7, 1], gap="large")' in home
    assert "WORKSPACE STATUS" in home
    assert "Ready for local analysis" in home
    assert "Current apparatus" not in home


def test_removed_status_copy_does_not_return_to_the_site():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Application v2 · prepared crop + manual" not in source
    assert 'st.warning("Automatic localization unavailable")' not in source
    assert "st.warning(status.block_reason)" not in source
    assert "Phase " not in source
    assert "CUDA" not in source


def test_prepared_upload_and_classify_ui_workflow():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/processed/wtbd_crops_v1/images/1_0.png"
    if not image.is_file():
        pytest.skip("local frozen Phase 3 crop payload is not available")
    app = open_analysis(testing.AppTest.from_file(str(APP_PATH)).run(timeout=30))
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/png").run(timeout=30)
    button(app, "Classify and add prepared crop").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[-1].value in set(HUMAN_LABELS.values())
    assert app.session_state["analysis_records"][0].region_id == "R1"
    score_data = chart_table(app.get("vega_lite_chart")[0])
    assert score_data.column_names == ["Category", "Model score"]
    assert score_data.num_rows == len(HUMAN_LABELS)
    assert all(row["Category"] in set(HUMAN_LABELS.values()) for row in score_data.to_pylist())
    next(item for item in app.radio if item.label == "Navigation").set_value("Compare Regions").run(timeout=30)
    assert {item.label for item in app.get("download_button")} >= {"Download JSON", "Download CSV"}
    comparison_tables = [chart_table(chart) for chart in app.get("vega_lite_chart")]
    comparison_data = next(table for table in comparison_tables if "Region" in table.column_names)
    assert comparison_data.column_names == ["Region", "Category", "Model score"]
    assert comparison_data.num_rows == len(HUMAN_LABELS)
    assert not app.exception


def test_manual_single_upload_and_classify_ui_workflow():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/raw/wtbd/WT blade defect dataset/JPEGImages/1.jpg"
    if not image.is_file():
        pytest.skip("local frozen WTBD source image payload is not available")
    app = open_analysis(testing.AppTest.from_file(str(APP_PATH)).run(timeout=30))
    next(item for item in app.radio if item.label == "Analysis mode").set_value("Manual single region").run(timeout=30)
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/jpeg").run(timeout=30)
    button(app, "Classify and add selected region").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[-1].value in set(HUMAN_LABELS.values())
    assert app.session_state["analysis_records"][0].selected_box is not None


def test_manual_multi_region_adds_stable_session_record():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/raw/wtbd/WT blade defect dataset/JPEGImages/1.jpg"
    if not image.is_file():
        pytest.skip("local frozen WTBD source image payload is not available")
    app = open_analysis(testing.AppTest.from_file(str(APP_PATH)).run(timeout=30))
    next(item for item in app.radio if item.label == "Analysis mode").set_value("Manual multi-region").run(timeout=30)
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/jpeg").run(timeout=30)
    button(app, "Add and classify region").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    saved = app.session_state["analysis_records"]
    assert len(saved) == 1
    assert saved[0].region_id == "R1"
    assert saved[0].mode == "manual_multi_region"


@pytest.mark.parametrize("page", ["Research Results", "Detection Readiness", "About"])
def test_read_only_pages_render_without_errors(page):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.radio[0].set_value(page).run(timeout=30)
    assert not app.exception
    assert not app.error
    rendered = "\n".join(element.value for element in app.markdown)
    assert "Phase " not in rendered
    assert "CUDA" not in rendered
    assert "unavailable" not in rendered.lower()
