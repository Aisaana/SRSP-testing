# tests/test_model.py
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_data_file_exists():
    """Проверка: файл данных существует"""
    assert os.path.exists('commits_dataset.csv'), "Dataset file not found"

def test_data_loading():
    """Проверка: данные загружаются корректно"""
    df = pd.read_csv('commits_dataset.csv')
    assert 'is_bug' in df.columns, "Column 'is_bug' not found"
    assert len(df) > 0, "Dataset is empty"
    assert len(df) >= 100, "Dataset too small"

def test_model_creation():
    """Проверка: модель создаётся"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    assert model is not None

def test_model_prediction():
    """Проверка: модель делает предсказания"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = [[100, 50, 10, 12, 0, 5]]
    model.fit(X, [0])
    prediction = model.predict(X)
    assert len(prediction) == 1