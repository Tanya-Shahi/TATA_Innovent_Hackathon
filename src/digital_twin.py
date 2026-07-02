import torch
import numpy as np
import pickle
import json
import os
from datetime import datetime
from model import LSTMModel


class EngineDigitalTwin:
    """
    Digital Twin for a single aircraft engine.
    Maintains state, predicts RUL, and generates alerts.
    """

    def __init__(self, engine_id, model, scaler, window_size=30):
        self.engine_id = engine_id
        self.model = model
        self.scaler = scaler
        self.window_size = window_size

        # State
        self.sensor_history = []
        self.rul_history = []
        self.cycle_count = 0
        self.alerts = []
        self.current_rul = None
        self.health_score = 100.0
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def ingest(self, sensor_readings):
        """
        Feed one cycle of sensor data into the twin.
        sensor_readings: list or array of sensor values
        """
        self.sensor_history.append(sensor_readings)
        self.cycle_count += 1

        if len(self.sensor_history) >= self.window_size:
            self._update_rul()
            self._update_health_score()
            self._check_alerts()

        return self.get_state()

    def _update_rul(self):
        window = np.array(self.sensor_history[-self.window_size:])
        window_scaled = self.scaler.transform(window)
        tensor = torch.FloatTensor(window_scaled).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            rul = self.model(tensor).item()

        self.current_rul = max(0, round(rul, 2))
        self.rul_history.append({
            'cycle': self.cycle_count,
            'rul': self.current_rul
        })

    def _update_health_score(self):
        if self.current_rul is not None:
            self.health_score = round(min(100, (self.current_rul / 125) * 100), 2)

    def _check_alerts(self):
        if self.current_rul is None:
            return

        if self.current_rul <= 10:
            self._add_alert("CRITICAL", f"Engine {self.engine_id} RUL critically low: {self.current_rul} cycles. Immediate maintenance required.")
        elif self.current_rul <= 30:
            self._add_alert("WARNING", f"Engine {self.engine_id} RUL low: {self.current_rul} cycles. Schedule maintenance soon.")
        elif self.current_rul <= 50:
            self._add_alert("CAUTION", f"Engine {self.engine_id} RUL: {self.current_rul} cycles. Monitor closely.")

    def _add_alert(self, level, message):
        alert = {
            'level': level,
            'message': message,
            'cycle': self.cycle_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if not self.alerts or self.alerts[-1]['message'] != message:
            self.alerts.append(alert)

    def get_state(self):
        return {
            'engine_id': self.engine_id,
            'cycle': self.cycle_count,
            'current_rul': self.current_rul,
            'health_score': self.health_score,
            'status': self._get_status(),
            'latest_alert': self.alerts[-1] if self.alerts else None,
            'rul_history': self.rul_history[-20:]
        }

    def _get_status(self):
        if self.current_rul is None:
            return "INITIALIZING"
        elif self.current_rul <= 10:
            return "CRITICAL"
        elif self.current_rul <= 30:
            return "WARNING"
        elif self.current_rul <= 50:
            return "CAUTION"
        else:
            return "HEALTHY"


class FleetDigitalTwin:
    """
    Manages digital twins for an entire fleet of engines.
    """

    def __init__(self, model_path='../models/lstm_rul.pt',
                 scaler_path='../models/scaler.pkl',
                 config_path='../models/model_config.json'):

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load model
        self.model = LSTMModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.window_size = config['window_size']
        self.engines = {}
        print(f"Fleet Digital Twin initialized with config: {config}")

    def add_engine(self, engine_id):
        if engine_id not in self.engines:
            self.engines[engine_id] = EngineDigitalTwin(
                engine_id=engine_id,
                model=self.model,
                scaler=self.scaler,
                window_size=self.window_size
            )
        return self.engines[engine_id]

    def ingest(self, engine_id, sensor_readings):
        if engine_id not in self.engines:
            self.add_engine(engine_id)
        return self.engines[engine_id].ingest(sensor_readings)

    def get_fleet_status(self):
        fleet = []
        for engine_id, twin in self.engines.items():
            fleet.append(twin.get_state())
        return sorted(fleet, key=lambda x: (x['current_rul'] or 999))

    def get_critical_engines(self):
        return [e for e in self.get_fleet_status()
                if e['status'] in ['CRITICAL', 'WARNING']]

    def get_fleet_summary(self):
        statuses = [t._get_status() for t in self.engines.values()]
        return {
            'total_engines': len(self.engines),
            'healthy': statuses.count('HEALTHY'),
            'caution': statuses.count('CAUTION'),
            'warning': statuses.count('WARNING'),
            'critical': statuses.count('CRITICAL'),
            'initializing': statuses.count('INITIALIZING')
        }