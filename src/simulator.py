import pandas as pd
import numpy as np
import time


class CMAPSSSimulator:
    """
    Replays NASA CMAPSS data cycle by cycle
    to simulate live engine telemetry streaming.
    """

    def __init__(self, data_path='../data/raw/train_FD001.txt'):
        columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
                  [f'sensor{i}' for i in range(1, 22)]

        df = pd.read_csv(data_path, sep='\s+', header=None, names=columns)

        # Drop same columns as training
        drop_cols = ['sensor1', 'sensor5', 'sensor6', 'sensor10',
                     'sensor16', 'sensor18', 'sensor19',
                     'setting1', 'setting2', 'setting3']
        df.drop(columns=drop_cols, inplace=True)

        self.df = df
        self.feature_cols = [c for c in df.columns
                             if c not in ['engine_id', 'cycle']]
        self.engine_ids = df['engine_id'].unique()
        print(f"Simulator loaded: {len(self.engine_ids)} engines, "
              f"{len(self.feature_cols)} features")

    def get_engine_data(self, engine_id):
        eng = self.df[self.df['engine_id'] == engine_id]
        return eng[self.feature_cols].values.tolist()

    def stream_engine(self, engine_id, delay=0.0):
        """
        Generator: yields one cycle of sensor data at a time.
        Set delay > 0 for real-time simulation.
        """
        data = self.get_engine_data(engine_id)
        for cycle_data in data:
            if delay > 0:
                time.sleep(delay)
            yield cycle_data

    def get_sample_engines(self, n=5):
        return list(self.engine_ids[:n])