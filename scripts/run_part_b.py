"""Reproduce the completed Part B stages from the project root.

Run:

    python scripts/run_part_b.py

The final Word/PDF report and Streamlit app are separate delivery steps.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.build_funds import main as build_funds  # noqa: E402
from scripts.build_fund_exhibits import main as build_fund_exhibits  # noqa: E402
from scripts.build_fusion import main as build_fusion  # noqa: E402
from scripts.build_sentiment import main as build_sentiment  # noqa: E402
from scripts.plot_sentiment import main as plot_sentiment  # noqa: E402
from scripts.plot_fusion import main as plot_fusion  # noqa: E402
from scripts.prepare_part_b_data import main as prepare_part_b_data  # noqa: E402


def main() -> None:
    print("=== REPRODUCING COMPLETED PART B STAGES ===")
    prepare_part_b_data()
    build_funds()
    build_fund_exhibits()
    build_sentiment()
    plot_sentiment()
    build_fusion()
    plot_fusion()
    print("\nCompleted: data, funds, fund exhibits, sentiment, and fusion.")
    print("Completed app: streamlit run streamlit_app.py")
    print("Review draft: report/report.docx")
    print("Pending: student rewrite/confirmation, final PDF, deployment, and hand-in QA.")


if __name__ == "__main__":
    main()
