import os
import pandas as pd


def test_movies_dataset_exists():
    assert os.path.exists('archive (6)/movies.csv')


def test_movies_dataset_loads():
    df = pd.read_csv('archive (6)/movies.csv', nrows=5)
    assert not df.empty
