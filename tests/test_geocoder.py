import pandas as pd
import pytest
from sloyka.src.geocoder import Geocoder

osm_city_level: int = 5
osm_city_name: str = "Санкт-Петербург"

@pytest.mark.parametrize(
    "input_address,geocode_result",
    [
        ("возле дома на Итальянской 17 постоянно мусорят!!!", "Итальянской 17"),
    ],
)
def test_geolocator(input_address, geocode_result):
    result = Geocoder().extract_ner_street(input_address)
    assert result.loc[0] == geocode_result



@pytest.fixture
def sample_dataframe():
    s_data = {
        "Текст комментария": [
            "Рубинштейна 25 дворовую территорию уберите, где работники?"
        ]
    }
    return pd.DataFrame(s_data)


def test_run_function(sample_dataframe):
    instance = Geocoder(osm_city_name=osm_city_name, osm_city_level=osm_city_level)

    result_df = instance.run(sample_dataframe)

    assert result_df.loc[0, "Street"] == "рубинштейна"
    assert result_df.loc[0, "Numbers"] == "25"