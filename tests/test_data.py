import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import os


def test_data_files_exist():
    required_files = [
        '../data/raw/train_FD001.txt',
        '../data/raw/test_FD001.txt',
        '../data/raw/RUL_FD001.txt'
    ]
    for f in required_files:
        assert os.path.exists(f), f"Missing file: {f}"
    print("test_data_files_exist passed")


def test_data_loading():
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]

    df = pd.read_csv('../data/raw/train_FD001.txt',
                     sep='\s+', header=None, names=columns)

    assert df.shape[1] == 26, f"Expected 26 columns, got {df.shape[1]}"
    assert df['engine_id'].nunique() == 100, "Expected 100 engines"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    print(f"test_data_loading passed | Shape: {df.shape}")


def test_rul_calculation():
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]

    df = pd.read_csv('../data/raw/train_FD001.txt',
                     sep='\s+', header=None, names=columns)

    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']

    assert df['RUL'].min() == 0, "Minimum RUL should be 0"
    assert df['RUL'].max() > 0, "Maximum RUL should be > 0"
    print(f"test_rul_calculation passed | RUL range: 0 to {df['RUL'].max()}")


def test_sensor_count():
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]

    df = pd.read_csv('../data/raw/train_FD001.txt',
                     sep='\s+', header=None, names=columns)

    sensor_cols = [c for c in df.columns if 'sensor' in c]
    assert len(sensor_cols) == 21, f"Expected 21 sensors, got {len(sensor_cols)}"
    print(f"test_sensor_count passed | Sensors: {len(sensor_cols)}")


if __name__ == "__main__":
    test_data_files_exist()
    test_data_loading()
    test_rul_calculation()
    test_sensor_count()
    print("\nAll data tests passed!")