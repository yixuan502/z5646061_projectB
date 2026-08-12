"""Network-free tests for scoring, equal-ticker aggregation, and signal timing."""

import numpy as np
import pandas as pd
import pytest

from src.sentiment import (
    sector_sentiment_index,
    score_headlines,
    ticker_day_sentiment,
)


class ExactTextAnalyzer:
    """Deterministic test double that records untouched headline strings."""

    def __init__(self, scores):
        self.scores = scores
        self.seen = []

    def polarity_scores(self, text):
        self.seen.append(text)
        compound = self.scores[text]
        if compound > 0:
            return {"neg": 0.0, "neu": 0.4, "pos": 0.6, "compound": compound}
        if compound < 0:
            return {"neg": 0.6, "neu": 0.4, "pos": 0.0, "compound": compound}
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}


def test_scoring_preserves_case_punctuation_and_labels():
    text = ["PROFIT beats estimates!!!", "Risk rises?"]
    analyzer = ExactTextAnalyzer({text[0]: 0.8, text[1]: -0.4})
    headlines = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
            "ticker": ["A", "A"],
            "sector": ["Tech", "Tech"],
            "title": text,
        }
    )

    scored = score_headlines(headlines, analyzer=analyzer)

    assert analyzer.seen == text
    assert scored["title"].tolist() == text
    assert scored["sentiment_label"].tolist() == ["positive", "negative"]


def test_sector_index_equal_weights_tickers_not_headlines():
    date = pd.Timestamp("2023-01-02")
    headlines = pd.DataFrame(
        {
            "trading_date": [date, date, date],
            "ticker": ["A", "A", "B"],
            "sector": ["Tech", "Tech", "Tech"],
            "title": ["A1", "A2", "B1"],
        }
    )
    analyzer = ExactTextAnalyzer({"A1": 1.0, "A2": 1.0, "B1": -1.0})
    ticker_scores = ticker_day_sentiment(score_headlines(headlines, analyzer))
    index = sector_sentiment_index(
        ticker_scores,
        pd.DatetimeIndex([date]),
        pd.DataFrame({"ticker": ["A", "B"], "sector": ["Tech", "Tech"]}),
    )

    row = index.iloc[0]
    assert row["raw_sector_sentiment"] == pytest.approx(0.0)
    assert row["headline_weighted_sentiment"] == pytest.approx(1.0 / 3.0)
    assert row["active_tickers"] == 2
    assert row["headline_count"] == 3


def test_no_news_is_missing_and_lag_uses_prior_trading_day():
    dates = pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-10"])
    headlines = pd.DataFrame(
        {
            "trading_date": [dates[0], dates[1]],
            "ticker": ["A", "A"],
            "sector": ["Tech", "Tech"],
            "title": ["Friday", "Weekend mapped to Monday"],
        }
    )
    analyzer = ExactTextAnalyzer({"Friday": 0.2, "Weekend mapped to Monday": 0.7})
    ticker_scores = ticker_day_sentiment(score_headlines(headlines, analyzer))
    mapping = pd.DataFrame(
        {"ticker": ["A", "B", "C"], "sector": ["Tech", "Tech", "Utilities"]}
    )
    index = sector_sentiment_index(ticker_scores, dates, mapping)

    tech = index[index["sector"].eq("Tech")].set_index("date")
    utilities = index[index["sector"].eq("Utilities")].set_index("date")
    assert tech.loc[dates[1], "lagged_sector_sentiment"] == pytest.approx(0.2)
    assert tech.loc[dates[2], "lagged_sector_sentiment"] == pytest.approx(0.7)
    assert tech.loc[dates[2], "signal_source_date"] == dates[1]
    assert np.isnan(utilities.loc[dates[1], "raw_sector_sentiment"])
    assert utilities.loc[dates[1], "raw_sentiment_label"] == "no_news"
    assert utilities.loc[dates[1], "ticker_coverage_ratio"] == 0.0
    assert not utilities.loc[dates[1], "lagged_signal_available"]


def test_duplicate_headline_keys_are_rejected():
    date = pd.Timestamp("2023-01-02")
    duplicate = pd.DataFrame(
        {
            "date": [date, date],
            "trading_date": [date, date],
            "ticker": ["A", "A"],
            "sector": ["Tech", "Tech"],
            "title": ["Same", "Same"],
        }
    )
    with pytest.raises(ValueError, match="duplicate headline keys"):
        score_headlines(duplicate, ExactTextAnalyzer({"Same": 0.0}))


def test_same_title_on_different_source_dates_survives_alignment_key_check():
    trading_date = pd.Timestamp("2023-01-09")
    source_dates = pd.to_datetime(["2023-01-08", "2023-01-09"])
    repeated = pd.DataFrame(
        {
            "date": source_dates,
            "trading_date": [trading_date, trading_date],
            "ticker": ["A", "A"],
            "sector": ["Tech", "Tech"],
            "title": ["Syndicated title", "Syndicated title"],
        }
    )

    scored = score_headlines(
        repeated,
        ExactTextAnalyzer({"Syndicated title": 0.1}),
    )

    assert len(scored) == 2
    assert scored["date"].nunique() == 2
