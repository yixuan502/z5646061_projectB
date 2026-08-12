"""App-level smoke tests for the four-step Streamlit investor journey."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_default_app_renders_full_investor_journey_without_exceptions():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Compare Funds",
        "Fund Fact Sheet",
        "Allocation Lab",
        "News & Innovation",
    ]
    assert {item.label for item in app.selectbox} == {"Choose a fund"}
    assert {item.label for item in app.radio} == {"Fusion fund family"}
    assert len(app.metric) >= 20
    assert len(app.dataframe) >= 4


def test_fact_sheet_fund_selection_reruns_cleanly():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    selector = next(item for item in app.selectbox if item.label == "Choose a fund")
    selector.set_value("Crypto Minimum Variance")
    app.run()

    assert not app.exception
    assert selector.value == "Crypto Minimum Variance"
