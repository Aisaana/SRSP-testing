import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def test_data_loading():
    df = pd.read_csv('commits_dataset.csv')
    assert 'is_bug' in df.columns
    assert len(df) > 0

def test_model_creation():
    model = RandomForestClassifier(n_estimators=10)
    assert model is not None