import pandas as pd
import pytest

from northstar.data.fangraphs import fetch_fangraphs_batting


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "data": [
                {
                    "xMLBAMID": "1001",
                    "PlayerName": "Player One",
                    "PA": 620,
                    "OPS": 0.950,
                    "wOBA": 0.400,
                    "wRC+": 150,
                },
                {
                    "xMLBAMID": "1002",
                    "PlayerName": "Player Two",
                    "PA": 250,
                    "OPS": 0.780,
                    "wOBA": 0.330,
                    "wRC+": 105,
                },
            ]
        }


class FakeSession:
    def get(self, *args, **kwargs):
        return FakeResponse()


def test_fetch_fangraphs_batting_normalizes_and_filters():
    result = fetch_fangraphs_batting(
        season=2025,
        min_pa=300,
        session=FakeSession(),
    )

    assert isinstance(result, pd.DataFrame)
    assert result["xMLBAMID"].tolist() == [1001]
    assert result["Name"].tolist() == ["Player One"]
    assert result["wRC+"].tolist() == [150]


def test_fetch_fangraphs_batting_validates_min_pa():
    with pytest.raises(ValueError, match="min_pa"):
        fetch_fangraphs_batting(season=2025, min_pa=0, session=FakeSession())
