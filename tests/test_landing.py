"""The case carousel changes all previews together and wraps both ways."""
from pathlib import Path
import json

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_landing_cases_stay_synchronized_and_wrap():
    cases = json.loads((ROOT / "app/assets/landing/cases.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)

    def check(index):
        assert not app.exception
        rendered = "\n".join(item.value for item in app.markdown)
        assert f'EXAMPLE {index + 1} OF {len(cases)}' in rendered
        assert f'CASE {index + 1:02d} / REGION PROPOSALS' in rendered
        assert 'href="#classification-workspace"' in rendered
        return next(item.value for item in app.markdown if '<section class="landing">' in item.value)

    initial = check(0)
    preview_container = next(block for block in app.get("flex_container") if block.proto.id.endswith("-landing_preview"))
    assert {button.key for button in preview_container.button} == {"landing_previous", "landing_next"}
    app.button(key="landing_next").click().run(timeout=30)
    assert check(1) != initial
    app.button(key="landing_previous").click().run(timeout=30)
    assert check(0) == initial
    app.button(key="landing_previous").click().run(timeout=30)
    check(len(cases) - 1)
    app.button(key="landing_next").click().run(timeout=30)
    assert check(0) == initial
