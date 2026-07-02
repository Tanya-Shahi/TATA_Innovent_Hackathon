import sys
sys.path.append('../src')

import torch
import numpy as np
import pickle
from digital_twin import EngineDigitalTwin, FleetDigitalTwin
from model import LSTMModel


def test_engine_twin_initialization():
    model = LSTMModel(input_size=14)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    twin = EngineDigitalTwin(
        engine_id=1,
        model=model,
        scaler=scaler,
        window_size=30
    )

    assert twin.engine_id == 1
    assert twin.cycle_count == 0
    assert twin.current_rul is None
    assert twin.health_score == 100.0
    print("test_engine_twin_initialization passed")


def test_engine_twin_ingest():
    model = LSTMModel(input_size=14)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    twin = EngineDigitalTwin(
        engine_id=1,
        model=model,
        scaler=scaler,
        window_size=30
    )

    for _ in range(35):
        dummy_data = np.random.rand(14).tolist()
        state = twin.ingest(dummy_data)

    assert twin.cycle_count == 35
    assert twin.current_rul is not None
    assert 0 <= twin.health_score <= 100
    print(f"test_engine_twin_ingest passed | RUL: {twin.current_rul}")


def test_fleet_twin_initialization():
    fleet = FleetDigitalTwin(
        model_path='../models/lstm_rul.pt',
        scaler_path='../models/scaler.pkl',
        config_path='../models/model_config.json'
    )

    assert fleet.model is not None
    assert fleet.scaler is not None
    assert fleet.window_size == 30
    assert len(fleet.engines) == 0
    print("test_fleet_twin_initialization passed")


def test_fleet_add_engine():
    fleet = FleetDigitalTwin(
        model_path='../models/lstm_rul.pt',
        scaler_path='../models/scaler.pkl',
        config_path='../models/model_config.json'
    )

    fleet.add_engine(1)
    fleet.add_engine(2)
    fleet.add_engine(3)

    assert len(fleet.engines) == 3
    print(f"test_fleet_add_engine passed | Engines: {len(fleet.engines)}")


def test_alert_system():
    model = LSTMModel(input_size=14)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    twin = EngineDigitalTwin(
        engine_id=99,
        model=model,
        scaler=scaler,
        window_size=30
    )

    twin.current_rul = 5
    twin.cycle_count = 50
    twin._check_alerts()

    assert len(twin.alerts) > 0
    assert twin.alerts[-1]['level'] == 'CRITICAL'
    print(f"test_alert_system passed | Alert: {twin.alerts[-1]['level']}")


def test_fleet_summary():
    fleet = FleetDigitalTwin(
        model_path='../models/lstm_rul.pt',
        scaler_path='../models/scaler.pkl',
        config_path='../models/model_config.json'
    )

    fleet.add_engine(1)
    summary = fleet.get_fleet_summary()

    assert 'total_engines' in summary
    assert 'healthy'       in summary
    assert 'critical'      in summary
    assert 'warning'       in summary
    assert 'caution'       in summary
    print(f"test_fleet_summary passed | Summary: {summary}")


if __name__ == "__main__":
    test_engine_twin_initialization()
    test_engine_twin_ingest()
    test_fleet_twin_initialization()
    test_fleet_add_engine()
    test_alert_system()
    test_fleet_summary()
    print("\nAll digital twin tests passed!")