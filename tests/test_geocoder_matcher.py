import pytest
import pandas as pd
import re
from sloyka.src.geocoder import Geocoder  # Импортируйте новый класс

# Фикстура для создания DataFrame
@pytest.fixture
def test_df_groups():
    data = {
        'group_name': ['Пушкин'],
        'Текст комментария': ['Рубинштейна 25 дворовую территорию уберите, где работники?']
    }
    df_groups = pd.DataFrame(data)
    return df_groups

# Тест для функции run
def test_run(test_df_groups: pd.DataFrame):
    osm_id = 338635
    tags = {"admin_level": ["8"]}
    date = "2024-04-22T00:00:00Z"
    osm_city_level: int = 5
    osm_city_name: str = "Санкт-Петербург"
    
    instance = Geocoder(osm_city_name=osm_city_name, osm_city_level=osm_city_level)
    result_df = instance.run(osm_id, tags, date, test_df_groups)
    
    assert any(result_df['territory'].apply(lambda x: bool(re.search(r'Пушкин', x))))
    assert result_df.loc[0, "Street"] == "рубинштейна"
    assert result_df.loc[0, "Numbers"] == "25"
